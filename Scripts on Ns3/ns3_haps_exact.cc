/* ============================================================
 * NS-3 HAPS Simulation — generates NTNData_HAPS.csv
 * 16 columns matching uploaded file exactly
 * Target: 11,980 rows, 20 UEs, 6000s, 10s interval
 * Channel: 3GPP TR 38.811, 2 GHz, 20 km altitude
 * ============================================================ */
#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/mobility-module.h"
#include <fstream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <sys/stat.h>
using namespace ns3;
NS_LOG_COMPONENT_DEFINE("HapsExact");
struct UeState{double x,y,prev_snr,rain_rate,rain_dur;bool in_rain;double beam_id;};
static std::ofstream g_csv;static uint32_t g_rows=0;static std::vector<UeState> g_ues;
double PL_HAPS(double d2,double alt=20000,double freq=2e9){
    double d3=std::sqrt(d2*d2+alt*alt);
    double lambda=3e8/freq;
    double fspl=20*std::log10(4*M_PI*d3/lambda);
    double el=std::atan2(alt,std::max(1.0,d2))*180/M_PI;
    double p_los=(el>=50)?1.0:(el>=30)?0.95:(el>=15)?0.85:0.75;
    double sf=4*((rand()%200)-100)/100.0;
    return p_los*(fspl+11+sf)+(1-p_los)*(fspl+23+sf);
}
double ComputeSNR(double d2,double rain_db){
    double pl=PL_HAPS(d2);
    double noise=10*std::log10(1.38e-23*290*200e6)+30+9;
    return 10+30-pl+15-noise-rain_db;
}
void Collect(uint32_t n,double iv,double dur,double haps_speed){
    double now=Simulator::Now().GetSeconds();
    double hx=haps_speed*now*std::cos(0.001*now);
    double hy=haps_speed*now*std::sin(0.001*now);
    for(uint32_t i=0;i<n;i++){
        UeState& s=g_ues[i];
        double dx=s.x-hx,dy=s.y-hy;
        double d2=std::sqrt(dx*dx+dy*dy);
        double d3=std::sqrt(d2*d2+20000.0*20000.0);
        double el=std::atan2(20000.0,std::max(1.0,d2))*180/M_PI;
        double az=std::atan2(dy,dx)*180/M_PI; if(az<0)az+=360;
        if(!s.in_rain&&(rand()%1000)<8){s.in_rain=true;s.rain_rate=5+(rand()%35);s.rain_dur=0;}
        if(s.in_rain){s.rain_dur+=iv;if(s.rain_dur>600||(rand()%100)<5)s.in_rain=false;}
        if(!s.in_rain)s.rain_rate=0;
        double rain_db=0;
        if(s.rain_rate>0.1&&el>5){double path=5/std::sin(el*M_PI/180);rain_db=0.0367*std::pow(s.rain_rate,1.0)*path;}
        double snr=ComputeSNR(d2,rain_db);
        double sinr=snr-(2+std::fmod(s.beam_id,3))+((rand()%20)-10)/10.0;
        double rssi=-85+snr*0.5+((rand()%20)-10)/10.0;
        double ber=(sinr>20)?1e-7:(sinr>12)?1e-5:(sinr>6)?1e-4:(sinr>0)?1e-3:0.02;
        ber=std::max(0.0,ber+std::abs(((rand()%100)-50)*ber*0.02));
        double tp=std::max(0.0,200*std::log2(1+std::pow(10,sinr/10))*0.65);
        double prop=(d3/3e8)*1000;
        double lat=prop*2+((rand()%10)/10.0);
        double pl=std::max(0.0,std::min(49.0,(sinr<0)?25+(rand()%25):(sinr<5)?3+(rand()%8):(rand()%20)/10.0));
        double lqi=std::max(0.0,std::min(100.0,(sinr+10)*3.5));
        double se=std::max(0.0,std::log2(1+std::pow(10,sinr/10)));
        double dop=9.4;
        g_csv<<std::fixed<<"UE_"<<std::setfill('0')<<std::setw(3)<<i<<","
             <<std::setprecision(2)<<d3/1000<<","
             <<std::setprecision(3)<<snr<<","<<sinr<<","<<rssi<<","
             <<std::setprecision(8)<<ber<<","
             <<std::setprecision(3)<<tp<<","<<lat<<","<<pl<<","
             <<std::setprecision(1)<<dop<<","
             <<std::setprecision(4)<<prop<<","
             <<std::setprecision(2)<<s.rain_rate<<","<<rain_db<<","
             <<std::setprecision(3)<<lqi<<","<<se<<","
             <<"HAPS\n";
        g_rows++;
    }
    g_csv.flush();
    static int lp=-1;int pct=(int)(now/dur*100);
    if(pct/10>lp/10){lp=pct;std::cout<<"  "<<pct<<"%  Rows:"<<g_rows<<"\n";}
    if(now+iv<=dur)Simulator::Schedule(Seconds(iv),&Collect,n,iv,dur,haps_speed);
}
int main(int argc,char* argv[]){
    uint32_t n=20;double dur=6000,iv=10;
    CommandLine cmd;cmd.AddValue("nUes","UEs",n);cmd.AddValue("duration","s",dur);cmd.Parse(argc,argv);
    mkdir("data",0755);mkdir("data/sims",0755);mkdir("data/sims/haps-exact",0755);
    g_csv.open("data/sims/haps-exact/NTNData_HAPS.csv");
    g_csv<<"UE_ID,distance_km,SNR_dB,SINR_dB,RSSI_dBm,BER,Throughput_Mbps,Latency_ms,"
         <<"Packet_Loss_pct,Doppler_Hz,Propagation_Delay_ms,Rain_Rate_mmhr,Rain_Fade_dB,"
         <<"Link_Quality_Index,Spectral_Efficiency_bps_hz,network_type\n";
    g_ues.resize(n);srand(42);
    for(uint32_t i=0;i<n;i++){
        g_ues[i].x=(rand()%300000)-150000;g_ues[i].y=(rand()%300000)-150000;
        g_ues[i].prev_snr=13;g_ues[i].rain_rate=0;g_ues[i].in_rain=false;
        g_ues[i].rain_dur=0;g_ues[i].beam_id=i%48;
    }
    std::cout<<"NS-3 HAPS Simulation — NTNData_HAPS.csv\nUEs:"<<n<<" Duration:"<<dur<<"s\n";
    Simulator::Schedule(Seconds(iv),&Collect,n,iv,dur,20.0);
    Simulator::Stop(Seconds(dur));Simulator::Run();Simulator::Destroy();g_csv.close();
    std::cout<<"DONE! Rows:"<<g_rows<<"\nCSV: data/sims/haps-exact/NTNData_HAPS.csv\n";
    return 0;
}
