/* ============================================================
 * NS-3 5G NR Simulation — generates TN_Data_5G.csv
 * 16 columns matching uploaded file exactly
 * 3GPP TR 38.901 UMa, 3.5 GHz, 100 MHz
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
NS_LOG_COMPONENT_DEFINE("FiveGExact");
struct UeState{double x,y,speed_ms,heading_deg,prev_snr;int ho_count;double cell_id;};
static std::ofstream g_csv;static uint32_t g_rows=0;static std::vector<UeState> g_ues;
const double GNB_X[3]={0,433,-433};const double GNB_Y[3]={0,250,250};
double PL_UMa(double d,double f=3.5){
    if(d<1)d=1;
    double d3=std::sqrt(d*d+(25-1.5)*(25-1.5));
    double dbp=4*25*1.5*f*1e9/3e8;
    double pl=(d<=dbp)?28+22*std::log10(d3)+20*std::log10(f):
               28+40*std::log10(d3)+20*std::log10(f)-9*std::log10(dbp*dbp+552.25);
    double p_los=(d<=18)?1.0:18.0/d+std::exp(-d/63.0)*(1-18.0/d);
    pl+=p_los*(4*((rand()%200)-100)/100.0)+(1-p_los)*(6*((rand()%200)-100)/100.0);
    return pl;
}
double SNR_5G(double d){
    double noise=10*std::log10(1.38e-23*290*100e6)+30+9;
    return 46+15-PL_UMa(d)+0-noise;
}
int BestGNB(double x,double y){int b=0;double bs=-999;for(int i=0;i<3;i++){double dx=x-GNB_X[i],dy=y-GNB_Y[i],s=SNR_5G(std::sqrt(dx*dx+dy*dy));if(s>bs){bs=s;b=i;}}return b;}
void Collect(uint32_t n,double iv,double dur){
    double now=Simulator::Now().GetSeconds();
    for(uint32_t i=0;i<n;i++){
        UeState& s=g_ues[i];
        s.x+=s.speed_ms*iv*std::cos(s.heading_deg*M_PI/180);
        s.y+=s.speed_ms*iv*std::sin(s.heading_deg*M_PI/180);
        if(std::abs(s.x)>600){s.heading_deg=180-s.heading_deg;s.x=std::max(-600.0,std::min(600.0,s.x));}
        if(std::abs(s.y)>600){s.heading_deg=-s.heading_deg;s.y=std::max(-600.0,std::min(600.0,s.y));}
        if((rand()%100)<3)s.heading_deg=rand()%360;
        int gnb=BestGNB(s.x,s.y);
        double dx=s.x-GNB_X[gnb],dy=s.y-GNB_Y[gnb],dist=std::max(1.0,std::sqrt(dx*dx+dy*dy));
        double snr=SNR_5G(dist);
        double ici=3+5*((rand()%100)/100.0);
        double sinr=snr-ici+((rand()%20)-10)/10.0;
        double rssi=-70+snr*0.5+((rand()%20)-10)/10.0;
        double ber=(sinr>22)?1e-7:(sinr>14)?1e-5:(sinr>6)?1e-3:(sinr>0)?1e-2:0.1;
        ber=std::max(0.0,ber+std::abs(((rand()%100)-50)*ber*0.02));
        double tp=std::max(5.2,100*std::log2(1+std::pow(10,sinr/10))*0.75);
        double lat=1.0+0.3*((rand()%100)/100.0)+dist/1000*0.1;
        double pl=std::max(0.0,std::min(39.0,(sinr<0)?20+(rand()%20):(sinr<5)?2+(rand()%8):(rand()%30)/10.0));
        double lqi=std::max(0.0,std::min(100.0,(sinr+10)*3.5));
        double se=std::max(0.0,std::log2(1+std::pow(10,sinr/10)));
        double dop=s.speed_ms*std::cos(s.heading_deg*M_PI/180)*3.5e9/3e8;
        double prop=(dist/3e8)*1000;
        double rain_db=0;double rain_rate=0;
        if((rand()%1000)<3){rain_rate=2+(rand()%15);rain_db=0.001*rain_rate*(dist/1000);}
        bool ho=((int)s.cell_id!=gnb&&now>10);
        if(ho){s.ho_count++;s.cell_id=gnb;}
        g_csv<<std::fixed<<"UE_"<<std::setfill('0')<<std::setw(3)<<i<<","
             <<std::setprecision(4)<<dist/1000<<","
             <<std::setprecision(3)<<snr<<","<<sinr<<","<<rssi<<","
             <<std::setprecision(8)<<ber<<","
             <<std::setprecision(3)<<tp<<","<<lat<<","<<pl<<","
             <<std::setprecision(2)<<dop<<","
             <<std::setprecision(6)<<prop<<","
             <<std::setprecision(2)<<rain_rate<<","<<rain_db<<","
             <<std::setprecision(3)<<lqi<<","<<se<<","
             <<"5G_NR\n";
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
    mkdir("data",0755);mkdir("data/sims",0755);mkdir("data/sims/5g-exact",0755);
    g_csv.open("data/sims/5g-exact/TN_Data_5G.csv");
    g_csv<<"UE_ID,distance_km,SNR_dB,SINR_dB,RSSI_dBm,BER,Throughput_Mbps,Latency_ms,"
         <<"Packet_Loss_pct,Doppler_Hz,Propagation_Delay_ms,Rain_Rate_mmhr,Rain_Fade_dB,"
         <<"Link_Quality_Index,Spectral_Efficiency_bps_hz,network_type\n";
    g_ues.resize(n);srand(42);
    for(uint32_t i=0;i<n;i++){g_ues[i]={(double)((rand()%1200)-600),(double)((rand()%1200)-600),5+(rand()%25)/5.0,(double)(rand()%360),15.0,0,(double)BestGNB(0,0)};}
    std::cout<<"NS-3 5G NR Simulation — TN_Data_5G.csv\nUEs:"<<n<<" Duration:"<<dur<<"s\n";
    Simulator::Schedule(Seconds(iv),&Collect,n,iv,dur);
    Simulator::Stop(Seconds(dur));Simulator::Run();Simulator::Destroy();g_csv.close();
    std::cout<<"DONE! Rows:"<<g_rows<<"\nCSV: data/sims/5g-exact/TN_Data_5G.csv\n";
    return 0;
}
