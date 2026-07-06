/* ============================================================
 * NS-3 LEO Simulation — generates NTN_Data_LEO.csv
 * 16 columns matching uploaded file exactly
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
NS_LOG_COMPONENT_DEFINE("LeoExact");
struct UeState{double x,y,prev_snr,rain_rate,rain_dur;bool in_rain;double beam_id;};
static std::ofstream g_csv;static uint32_t g_rows=0;static std::vector<UeState> g_ues;
double ComputeSNR_LEO(double slant_km,double rain_db){
    double snr=8+12*std::sin(std::asin(std::min(1.0,550.0/slant_km)))-rain_db;
    snr+=1.5*((rand()%200)-100)/100.0;
    return snr;
}
void Collect(uint32_t n,double iv,double dur){
    double now=Simulator::Now().GetSeconds();
    double T=5760.0;
    for(uint32_t i=0;i<n;i++){
        UeState& s=g_ues[i];
        double phase=std::fmod(now/T*360+i*18,360)*M_PI/180;
        double slant=800+400*std::abs(std::sin(phase));
        double el=std::asin(std::min(1.0,550.0/slant))*180/M_PI;
        double dop=4738*std::cos(el*M_PI/180)*std::sin((rand()%360)*M_PI/180)/10.0;
        if(!s.in_rain&&(rand()%1000)<8){s.in_rain=true;s.rain_rate=5+(rand()%45);s.rain_dur=0;}
        if(s.in_rain){s.rain_dur+=iv;if(s.rain_dur>600||(rand()%100)<5)s.in_rain=false;}
        if(!s.in_rain)s.rain_rate=0;
        double rain_db=0;
        if(s.rain_rate>0.1&&el>5){double path=5/std::sin(el*M_PI/180);rain_db=0.0367*std::pow(s.rain_rate,1.0)*path;}
        double snr=ComputeSNR_LEO(slant,rain_db);
        double sinr=snr-(3+std::fmod(s.beam_id,3))+((rand()%20)-10)/10.0;
        double rssi=-86+snr*0.5+((rand()%20)-10)/10.0;
        double ber=(sinr>10)?1e-5:(sinr>5)?1e-4:(sinr>0)?1e-3:0.02;
        ber=std::max(0.0,ber+std::abs(((rand()%100)-50)*ber*0.02));
        double tp=std::max(0.0,500*std::log2(1+std::pow(10,sinr/10))*0.65);
        double prop=(slant*1e3/3e8)*1000;
        double lat=prop*2+((rand()%10)/10.0);
        double pl=std::max(0.1,std::min(59.0,(sinr<0)?30+(rand()%30):(sinr<5)?5+(rand()%15):0.1+(rand()%30)/10.0));
        double lqi=std::max(0.0,std::min(72.37,(sinr+10)*3.5));
        double se=std::max(0.0,std::log2(1+std::pow(10,sinr/10)));
        g_csv<<std::fixed<<"UE_"<<std::setfill('0')<<std::setw(3)<<i<<","<<"LEO,"
             <<std::setprecision(2)<<slant<<","
             <<std::setprecision(3)<<snr<<","<<sinr<<","<<rssi<<","
             <<std::setprecision(8)<<ber<<","
             <<std::setprecision(3)<<tp<<","<<lat<<","<<pl<<","
             <<std::setprecision(1)<<dop<<","
             <<std::setprecision(4)<<prop<<","
             <<std::setprecision(2)<<s.rain_rate<<","<<rain_db<<","
             <<std::setprecision(3)<<lqi<<","<<se<<"\n";
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
    mkdir("data",0755);mkdir("data/sims",0755);mkdir("data/sims/leo-exact",0755);
    g_csv.open("data/sims/leo-exact/NTN_Data_LEO.csv");
    g_csv<<"UE_ID,sat_type,distance_km,SNR_dB,SINR_dB,RSSI_dBm,BER,Throughput_Mbps,"
         <<"Latency_ms,Packet_Loss_pct,Doppler_Hz,Propagation_Delay_ms,"
         <<"Rain_Rate_mmhr,Rain_Fade_dB,Link_Quality_Index,Spectral_Efficiency_bps_hz\n";
    g_ues.resize(n);srand(42);
    for(uint32_t i=0;i<n;i++){g_ues[i]={0,0,7.3,0,0,false,(double)(i%24)};}
    std::cout<<"NS-3 LEO Simulation — NTN_Data_LEO.csv\nUEs:"<<n<<" Duration:"<<dur<<"s\n";
    Simulator::Schedule(Seconds(iv),&Collect,n,iv,dur);
    Simulator::Stop(Seconds(dur));Simulator::Run();Simulator::Destroy();g_csv.close();
    std::cout<<"DONE! Rows:"<<g_rows<<"\nCSV: data/sims/leo-exact/NTN_Data_LEO.csv\n";
    return 0;
}
