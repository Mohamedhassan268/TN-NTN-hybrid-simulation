/* ============================================================
 * NS-3 UAV Simulation — generates Data_UAV.csv
 * 18 columns: UE_ID, distance_km, SNR_dB, SINR_dB, RSSI_dBm,
 *   BER, Throughput_Mbps, Latency_ms, Packet_Loss_pct,
 *   Doppler_Hz, Propagation_Delay_ms, Rain_Rate_mmhr,
 *   Rain_Fade_dB, Link_Quality_Index,
 *   Spectral_Efficiency_bps_hz, altitude_m, speed_ms,
 *   network_type
 * Target: 11,980 rows, 20 UAVs, 6000s, 10s interval
 * Channel: 3GPP TR 36.777 A2G, 2.4 GHz
 *
 * INSTALL:
 *   cp ns3_uav_exact.cc \
 *     ~/ns3-sns3/ns-allinone-3.43/ns-3.43/scratch/uav-exact.cc
 * RUN:
 *   cd ~/ns3-sns3/ns-allinone-3.43/ns-3.43
 *   ./ns3 build
 *   ./ns3 run uav-exact
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
NS_LOG_COMPONENT_DEFINE("UavExact");
struct UeState{
    double x,y,altitude_m,speed_ms,heading_deg,prev_snr,rain_rate,rain_dur;
    bool in_rain; int ho_count; double beam_id;
    double wp_x[6],wp_y[6],wp_alt[6]; int wp_idx;
};
static std::ofstream g_csv; static uint32_t g_rows=0;
static std::vector<UeState> g_ues;
double PL_A2G(double dist_3d,double alt,double freq=2.4e9){
    double lambda=3e8/freq;
    double fspl=20*std::log10(4*M_PI*dist_3d/lambda);
    double dist_2d=std::sqrt(std::max(1.0,dist_3d*dist_3d-alt*alt));
    double el=std::atan2(alt,dist_2d)*180/M_PI;
    double p_los=(el>=60)?1.0:(el>=30)?0.9:(el>=15)?0.7:0.5;
    return p_los*(fspl+1)+( 1-p_los)*(fspl+21);
}
double ComputeSNR(double dist_3d,double alt,double rain_db){
    double pl=PL_A2G(dist_3d,alt);
    double noise=10*std::log10(1.38e-23*290*20e6)+30+7;
    double snr=30+15-pl+3-noise-rain_db;
    snr+=4*((rand()%200)-100)/100.0;
    return snr;
}
void UpdatePos(UeState& s,double dt){
    double tx=s.wp_x[s.wp_idx],ty=s.wp_y[s.wp_idx],ta=s.wp_alt[s.wp_idx];
    double dx=tx-s.x,dy=ty-s.y,da=ta-s.altitude_m;
    double dist=std::sqrt(dx*dx+dy*dy+da*da);
    if(dist<10){s.wp_idx=(s.wp_idx+1)%6;return;}
    s.speed_ms=10+(rand()%100)/10.0;
    double step=s.speed_ms*dt;
    s.x+=(dx/dist)*step; s.y+=(dy/dist)*step;
    s.altitude_m=std::max(10.0,std::min(500.0,s.altitude_m+(da/dist)*step));
    s.heading_deg=std::atan2(dy,dx)*180/M_PI;
    if(s.heading_deg<0)s.heading_deg+=360;
}
void Collect(uint32_t n,double iv,double dur){
    double now=Simulator::Now().GetSeconds();
    for(uint32_t i=0;i<n;i++){
        UeState& s=g_ues[i];
        UpdatePos(s,iv);
        double d2=std::sqrt(s.x*s.x+s.y*s.y);
        double d3=std::sqrt(d2*d2+s.altitude_m*s.altitude_m);
        if(!s.in_rain&&(rand()%1000)<6){s.in_rain=true;s.rain_rate=2+(rand()%20);s.rain_dur=0;}
        if(s.in_rain){s.rain_dur+=iv;if(s.rain_dur>600||(rand()%100)<5)s.in_rain=false;}
        if(!s.in_rain)s.rain_rate=0;
        double rain_db=0;
        if(s.rain_rate>0.1){double dk=d3/1000;rain_db=0.0125*std::pow(s.rain_rate,1.31)*dk;}
        double snr=ComputeSNR(d3,s.altitude_m,rain_db);
        double sinr=snr-(2+std::fmod(s.beam_id,3))+((rand()%20)-10)/10.0;
        double rssi=-70+snr*0.5+((rand()%20)-10)/10.0;
        double ber=(sinr>25)?0.0:(sinr>16)?1e-6:(sinr>8)?1e-4:(sinr>0)?1e-2:0.1;
        ber=std::max(0.0,ber+std::abs(((rand()%100)-50)*ber*0.01));
        double tp=std::max(0.0,20*std::log2(1+std::pow(10,sinr/10))*0.7);
        double dop=(s.speed_ms*std::cos(s.heading_deg*M_PI/180)*2.4e9)/3e8;
        double prop=(d3/3e8)*1000;
        double lat=prop*2+1+((rand()%10)/10.0);
        double pl=std::max(0.0,std::min(3.0,(sinr<0)?30+(rand()%30):(sinr<5)?5+(rand()%10):0.1+(rand()%30)/10.0));
        double lqi=std::max(0.0,std::min(100.0,(sinr+10)*3.5));
        double se=std::max(0.0,std::log2(1+std::pow(10,sinr/10)));
        g_csv<<std::fixed<<"UE_"<<std::setfill('0')<<std::setw(3)<<i<<","
             <<std::setprecision(4)<<d3/1000<<","
             <<std::setprecision(3)<<snr<<","<<sinr<<","<<rssi<<","
             <<std::setprecision(8)<<ber<<","
             <<std::setprecision(3)<<tp<<","<<lat<<","<<pl<<","
             <<std::setprecision(1)<<dop<<","
             <<std::setprecision(4)<<prop<<","
             <<std::setprecision(2)<<s.rain_rate<<","<<rain_db<<","
             <<std::setprecision(3)<<lqi<<","<<se<<","
             <<std::setprecision(1)<<s.altitude_m<<","<<s.speed_ms<<","
             <<"UAV\n";
        g_rows++;
    }
    g_csv.flush();
    static int lp=-1;int pct=(int)(now/dur*100);
    if(pct/10>lp/10){lp=pct;std::cout<<"  "<<pct<<"%  Rows:"<<g_rows<<"\n";}
    if(now+iv<=dur)Simulator::Schedule(Seconds(iv),&Collect,n,iv,dur);
}
int main(int argc,char* argv[]){
    uint32_t n=20;double dur=6000,iv=10;
    CommandLine cmd;cmd.AddValue("nUes","UEs",n);cmd.AddValue("duration","s",dur);cmd.Parse(argc,argv);
    mkdir("data",0755);mkdir("data/sims",0755);mkdir("data/sims/uav-exact",0755);
    g_csv.open("data/sims/uav-exact/Data_UAV.csv");
    g_csv<<"UE_ID,distance_km,SNR_dB,SINR_dB,RSSI_dBm,BER,Throughput_Mbps,Latency_ms,"
         <<"Packet_Loss_pct,Doppler_Hz,Propagation_Delay_ms,Rain_Rate_mmhr,Rain_Fade_dB,"
         <<"Link_Quality_Index,Spectral_Efficiency_bps_hz,altitude_m,speed_ms,network_type\n";
    g_ues.resize(n);srand(42);
    for(uint32_t i=0;i<n;i++){
        g_ues[i].x=(rand()%4000)-2000;g_ues[i].y=(rand()%4000)-2000;
        g_ues[i].altitude_m=50+(rand()%450);g_ues[i].speed_ms=10+(rand()%10);
        g_ues[i].heading_deg=rand()%360;g_ues[i].prev_snr=15;
        g_ues[i].rain_rate=0;g_ues[i].in_rain=false;g_ues[i].rain_dur=0;
        g_ues[i].ho_count=0;g_ues[i].beam_id=i%12;g_ues[i].wp_idx=0;
        for(int w=0;w<6;w++){g_ues[i].wp_x[w]=(rand()%4000)-2000;
            g_ues[i].wp_y[w]=(rand()%4000)-2000;g_ues[i].wp_alt[w]=50+(rand()%450);}
    }
    std::cout<<"NS-3 UAV Simulation — Data_UAV.csv\nUEs:"<<n<<" Duration:"<<dur<<"s\n";
    Simulator::Schedule(Seconds(iv),&Collect,n,iv,dur);
    Simulator::Stop(Seconds(dur));Simulator::Run();Simulator::Destroy();g_csv.close();
    std::cout<<"DONE! Rows:"<<g_rows<<"\nCSV: data/sims/uav-exact/Data_UAV.csv\n";
    return 0;
}
