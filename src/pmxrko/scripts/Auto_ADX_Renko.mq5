//+------------------------------------------------------------------+
//|                            Auto ADX(barabashkakvn's edition).mq5 |
//|                                              David_Angelic Enegy |
//+------------------------------------------------------------------+
#property copyright "Pierre Mergaux"
#property version   "1.000"
//---

// Augmenté à 24 pour pouvoir stocker au moins 18 briques pour la régression
#define BRICK_SIZE 20 
// #define REG_PERIODS 18 // Périodes pour la régression linéaire

struct Renko
  {
   datetime          time;
   double            openr;
   double            open;
   double            high;
   double            low;
   double            close;
   double            closer;
  };

//--- input RENKO
input double   InpRenkoSize      = 37.5;      // renko size
input int      InpRegPeriods     = 6;        // periods for linear regression
input int      InpErWindow       = 6;        // window for Kaufman ER
input ENUM_TIMEFRAMES InpTimeFrame = PERIOD_H4;
//--- global variables
Renko bricks[];
double ExtRenkoSize;

string    m_name_ceil="ceil";
string    m_name_round="round";
string    m_name_floor="floor";
color             InpColorCeil            = clrDodgerBlue;  // Line color
ENUM_LINE_STYLE   InpStyleCeil            = STYLE_DASH;     // Line style
int               InpWidthCeil            = 1;              // Line width
color             InpColorRound           = clrMagenta;        // Line color
ENUM_LINE_STYLE   InpStyleRound           = STYLE_DASH;     // Line style
int               InpWidthRound           = 1;              // Line width
color             InpColorFloor           = clrRed;         // Line color
ENUM_LINE_STYLE   InpStyleFloor           = STYLE_DASH;     // Line style
int               InpWidthFloor           = 1;              // Line width
color             InpColorOther           = clrGray;

// Pré-déclaration des fonctions
void ChangeTrendEmptyPoints(datetime &time1,double &price1, datetime &time2,double &price2);
bool HLineCreate(const long chart_ID, const string name, const int sub_window, double price, const color clr, const ENUM_LINE_STYLE style, const int width, const bool back=false, const bool selection=true, const bool hidden=true, const long z_order=0);
bool HLineMove(const long chart_ID, const string name, double price);
bool HLineDelete(const long chart_ID, const string name);
bool tick2renko(MqlTick &df[], double step=10.0);
bool tick22renko(MqlTick &df[], int start, double step);
bool getTicks(MqlTick &ti[], int size);
void calculRenko(void);
void DrawRegressionLine(int periods);

//-------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//---
   ExtRenkoSize = InpRenkoSize;
   if(ObjectFind(0,m_name_ceil)<0)
      HLineCreate(0,m_name_ceil,0,0.0,InpColorCeil,InpStyleCeil,InpWidthCeil);
   if(ObjectFind(0,m_name_round)<0)
      HLineCreate(0,m_name_round,0,0.0,InpColorRound,InpStyleRound,InpWidthRound);
   if(ObjectFind(0,m_name_floor)<0)
      HLineCreate(0,m_name_floor,0,0.0,InpColorFloor,InpStyleFloor,InpWidthFloor);
   if(ObjectFind(0,"D1")<0)
      HLineCreate(0,"D1",0,0.0,InpColorOther,InpStyleRound,InpWidthRound);
   if(ObjectFind(0,"D2")<0)
      HLineCreate(0,"D2",0,0.0,InpColorOther,InpStyleRound,InpWidthRound);
   if(ObjectFind(0,"U1")<0)
      HLineCreate(0,"U1",0,0.0,InpColorOther,InpStyleRound,InpWidthRound);
   if(ObjectFind(0,"U2")<0)
      HLineCreate(0,"U2",0,0.0,InpColorOther,InpStyleRound,InpWidthRound);

   MqlTick tick[];
   bool rt = getTicks(tick, 3000000);
   if(rt == false)
     {
      Print("Error get_ticks");
      return(INIT_FAILED);
     }
   rt = tick2renko(tick, ExtRenkoSize);
   if(rt == false || ArraySize(bricks) == 0)
     {
      Print("Error init renkos");
      return(INIT_FAILED);
     }
   Print("RK size ", ArraySize(bricks));
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitia/lization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//---
   HLineDelete(0,m_name_ceil);
   HLineDelete(0,m_name_round);
   HLineDelete(0,m_name_floor);
   HLineDelete(0,"D1");
   HLineDelete(0,"D2");
   HLineDelete(0,"U1");
   HLineDelete(0,"U2");
   ObjectDelete(0, "Reg_Line_18"); // Supprime la ligne de régression
   ObjectDelete(0, "Reg_R2_Text"); // Supprime le texte du R2
   Comment("");
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool tick2renko(MqlTick &df[], double step=10.0)      
  {
   long timeopen;
   long stt;
   int t;
   double price = df[0].bid;
   double pprice = floor(price / step) * step;
   double nprice = pprice;
   double iprice;
   int n = 0;
   int mult;
   n = ArraySize(bricks);
   if(n != 0)
     {
      ArrayFree(bricks);
      n = 0;
     }
   if(ArrayResize(bricks, n+1, 50)<1)
     {
      PrintFormat("%s err alloc resize",_Symbol);
      return(false);
     }
   bricks[0].time = df[0].time;
   bricks[0].openr = nprice;
   bricks[0].open = price;
   bricks[0].high = price;
   bricks[0].low = price;
   bricks[0].close = price;
   bricks[0].closer = 0.0;
   n++;

   for(int i=1; i < ArraySize(df); i++)
     {
      price = df[i].bid;
      pprice = bricks[n-1].openr;
      if(n == 1)
        {
         if(price > pprice + step)
           {
            bricks[n-1].closer = pprice + step;
            mult = int(floor((price - pprice) / step));
            nprice = pprice + mult * step;
            iprice = pprice + step;
            timeopen = bricks[n-1].time;
            stt = long((df[i].time_msc - timeopen) / mult);
            t = 1;
            while(iprice < nprice)
              {
               ArrayResize(bricks, n+1, 50);
               bricks[n].time = datetime(timeopen + stt * t);
               bricks[n].openr = iprice;
               bricks[n].open = price;
               bricks[n].high = price;
               bricks[n].low = price;
               bricks[n].close = price;
               bricks[n].closer = iprice + step;
               n++;
               iprice += step;
               t++;
              }
            ArrayResize(bricks, n+1, 50);
            bricks[n].time = df[i].time;
            bricks[n].openr = nprice;
            bricks[n].open = price;
            bricks[n].high = price;
            bricks[n].low = price;
            bricks[n].close = price;
            bricks[n].closer = 0.0;
            n++;
           }
         else
            if(price < pprice - step)
              {
               bricks[n-1].closer = pprice - step;
               mult = int(floor((pprice - price) / step));
               nprice = pprice - mult * step;
               iprice = pprice - step;
               timeopen = bricks[n-1].time;
               stt = (df[i].time - timeopen) / mult;
               t = 1;
               while(iprice > nprice)
                 {
                  ArrayResize(bricks, n+1, 50);
                  bricks[n].time = datetime(timeopen + stt * t);
                  bricks[n].openr = iprice;
                  bricks[n].open = price;
                  bricks[n].high = price;
                  bricks[n].low = price;
                  bricks[n].close = price;
                  bricks[n].closer = iprice - step;
                  n++;
                  iprice -= step;
                  t++;
                 }
               ArrayResize(bricks, n+1, 50);
               bricks[n].time = df[i].time;
               bricks[n].openr = nprice;
               bricks[n].open = price;
               bricks[n].high = price;
               bricks[n].low = price;
               bricks[n].close = price;
               bricks[n].closer = 0.0;
               n++;
              }
            else
              {
               if(price > bricks[n-1].high)
                  bricks[n-1].high= price;
               else
                  if(price < bricks[n-1].low)
                     bricks[n-1].low = price;
               bricks[n-1].close = price;
              }
        }
      else
         return (tick22renko(df, i, step));
     }
   return (true);
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool tick22renko(MqlTick &df[], int start, double step)      
  {
   long timeopen;
   long stt;
   int t;
   int mult;
   double price;
   double pprice;
   double nprice;
   double iprice;
   double aprice;
   double cprice;
   int n = ArraySize(bricks);
   for(int i=start; i < ArraySize(df); i++)
     {
      price = df[i].bid;
      pprice = bricks[n-1].openr;
      aprice = bricks[n-2].openr;
      cprice = bricks[n-2].closer;
      if(aprice > cprice && price > pprice + step * 2)
        {
         pprice = pprice + step;
         bricks[n-1].openr = pprice;
         bricks[n-1].closer = pprice + step;
         mult = int(floor((price - pprice) / step));
         nprice = pprice + mult * step;
         iprice = pprice + step;
         timeopen = bricks[n-1].time;
         stt = (df[i].time_msc - timeopen) / mult;
         t = 1;
         while(iprice < nprice)
           {
            ArrayResize(bricks, n+1, 50);
            bricks[n].time = datetime(timeopen + stt * t);
            bricks[n].openr = iprice;
            bricks[n].open = price;
            bricks[n].high = price;
            bricks[n].low = price;
            bricks[n].close = price;
            bricks[n].closer = iprice + step;
            n++;
            iprice += step;
            t++;
           }
         ArrayResize(bricks, n+1, 50);
         bricks[n].time = df[i].time;
         bricks[n].openr = nprice;
         bricks[n].open = price;
         bricks[n].high = price;
         bricks[n].low = price;
         bricks[n].close = price;
         bricks[n].closer = 0.0;
         n++;
        }
      else
         if(aprice < cprice && price > pprice + step)
           {
            bricks[n-1].closer = pprice + step;
            mult = int(floor((price - pprice) / step));
            nprice = pprice + mult * step;
            iprice = pprice + step;
            timeopen = bricks[n-1].time;
            stt = (df[i].time_msc - timeopen) / mult;
            t = 1;
            while(iprice < nprice)
              {
               ArrayResize(bricks, n+1, 50);
               bricks[n].time = datetime(timeopen + stt * t);
               bricks[n].openr = iprice;
               bricks[n].open = price;
               bricks[n].high = price;
               bricks[n].low = price;
               bricks[n].close = price;
               bricks[n].closer = iprice + step;
               n++;
               iprice += step;
               t++;
              }
            ArrayResize(bricks, n+1, 50);
            bricks[n].time = df[i].time;
            bricks[n].openr = nprice;
            bricks[n].open = price;
            bricks[n].high = price;
            bricks[n].low = price;
            bricks[n].close = price;
            bricks[n].closer = 0.0;
            n++;
           }
         else
            if(aprice < cprice && price < pprice - step * 2)
              {
               pprice = pprice - step;
               bricks[n-1].openr = pprice;
               bricks[n-1].closer = pprice - step;
               mult = int(floor((pprice - price) / step));
               nprice = pprice - mult * step;
               iprice = pprice - step;
               timeopen = bricks[n-1].time;
               stt = (df[i].time - timeopen) / mult;
               t = 1;
               while(iprice > nprice)
                 {
                  ArrayResize(bricks, n+1, 50);
                  bricks[n].time = datetime(timeopen + stt * t);
                  bricks[n].openr = iprice;
                  bricks[n].open = price;
                  bricks[n].high = price;
                  bricks[n].low = price;
                  bricks[n].close = price;
                  bricks[n].closer = iprice - step;
                  n++;
                  iprice -= step;
                  t++;
                 }
               ArrayResize(bricks, n+1, 50);
               bricks[n].time = df[i].time;
               bricks[n].openr = nprice;
               bricks[n].open = price;
               bricks[n].high = price;
               bricks[n].low = price;
               bricks[n].close = price;
               bricks[n].closer = 0.0;
               n++;
              }
            else
               if(aprice > cprice && price < pprice - step)
                 {
                  bricks[n-1].closer = pprice - step;
                  mult = int(floor((pprice - price) / step));
                  nprice = pprice - mult * step;
                  iprice = pprice - step;
                  timeopen = bricks[n-1].time;
                  stt = (df[i].time_msc - timeopen) / mult;
                  t = 1;
                  while(iprice > nprice)
                    {
                     ArrayResize(bricks, n+1, 50);
                     bricks[n].time = datetime(timeopen + stt * t);
                     bricks[n].openr = iprice;
                     bricks[n].open = price;
                     bricks[n].high = price;
                     bricks[n].low = price;
                     bricks[n].close = price;
                     bricks[n].closer = iprice - step;
                     n++;
                     iprice -= step;
                     t++;
                    }
                  ArrayResize(bricks, n+1, 50);
                  bricks[n].time = df[i].time;
                  bricks[n].openr = nprice;
                  bricks[n].open = price;
                  bricks[n].high = price;
                  bricks[n].low = price;
                  bricks[n].close = price;
                  bricks[n].closer = 0.0;
                  n++;
                 }
               else
                 {
                  if(price > bricks[n-1].high)
                     bricks[n-1].high = price;
                  else
                     if(price < bricks[n-1].low)
                        bricks[n-1].low = price;
                  bricks[n-1].close = price;
                 }
     }
   return (true);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool getTicks(MqlTick &ti[], int size)
  {
   MqlDateTime actuel;
   datetime dactuel=TimeLocal(actuel);
   Print(dactuel);
   int num = CopyTicks(_Symbol, ti, COPY_TICKS_ALL, 0, size);
   Print("GT ", num, " sz ",size);
   if(num == -1)
     {
      return(false);
     }
   return(true);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool getTicksRange(MqlTick &ti[], ulong dateFrom)
  {
   int num = CopyTicksRange(_Symbol,ti,COPY_TICKS_ALL,dateFrom);
   if(num==-1)
     {
      return(false);
     }
   return(true);
  }

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
double bid = 0;
double ask = 0;
double renko2 = 0;
int schem[BRICK_SIZE];

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void OnTick()
  {
   static int taille = 1000;
   static MqlTick lticks[];
//---
   MqlTick lt[1];
   SymbolInfoTick(_Symbol,lt[0]);
   bid = lt[0].bid;
   ask = lt[0].ask;
   datetime dtk = lt[0].time;

   if(ArraySize(bricks) == 0 && taille > 0)
     {
      int n = ArraySize(lticks);
      if(n < taille)
        {
         ArrayResize(lticks,n+1,taille);
         lticks[n] = lt[0];
         return;
        }
      else
         if(n==taille)
           {
            tick2renko(lticks,ExtRenkoSize);
            if(ArraySize(bricks) < BRICK_SIZE+2)
              {
               taille += 100;
               return;
              }
            else
              {
               PrintFormat("Fin de test %d", taille);
               taille = 0;
              }
           }
     }
     else
      taille = 0;
      
   int bn = 0;
//------------------- si changement renko size
   if(tick22renko(lt,0, ExtRenkoSize) == false)
      return;
      
// ------------------------------------------------------------------       les bougies renko
   bn = ArraySize(bricks);
   if(bn < BRICK_SIZE+2)
      return;
   if(bn > BRICK_SIZE+10)
     {
      ArrayRemove(bricks, 0, bn - (BRICK_SIZE+3));
      bn = ArraySize(bricks);
     }
// ---
   for(int i=0; i<BRICK_SIZE-1; i++)
     {
      if(bricks[bn - BRICK_SIZE + i].openr < bricks[bn - BRICK_SIZE + i].closer)
         schem[i] = 1;
      else
         schem[i] = -1;
     }

   if(bricks[bn-1].openr - bid > ExtRenkoSize *  0.5)
      schem[BRICK_SIZE-1] = -1;
   else
      if(bid - bricks[bn-2].openr > ExtRenkoSize * 0.5)
         schem[BRICK_SIZE-1] = 1;
      else
         schem[BRICK_SIZE-1] = 0;

// ------------------------------------------------------------------       le schema renko
   renko2 = bricks[bn - 2].openr;
   calculRenko();
   
// ------------------------------------------------------------------       Regression
   DrawRegressionLine();
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void calculRenko(void)
  {
   short sRenko = (short)schem[BRICK_SIZE-2];
   double exosize = ExtRenkoSize*sRenko;
   HLineMove(0,m_name_ceil, renko2 + exosize);
   HLineMove(0,m_name_round, renko2);
   HLineMove(0, m_name_floor, renko2 - exosize);
   HLineMove(0,"D1", renko2+exosize*2);
   HLineMove(0,"D2", renko2+exosize*3);
   HLineMove(0,"U1", renko2-exosize*2);
   HLineMove(0,"U2", renko2-exosize*3);
  }

//+----------------------------------------------------------------------------------------------------+
//| Fonction qui calcule et dessine la régression linéaire sur N périodes (Chandeliers 1H)             |
//+----------------------------------------------------------------------------------------------------+
void DrawRegressionLine()
  {
   //Print("rates");
   // On récupère les chandeliers japonais 1 Heure (PERIOD_H1)
   // On s'assure de copier assez de bougies pour satisfaire à la fois la régression et le calcul de l'ER
   int max_window = MathMax(InpRegPeriods, InpErWindow);
   MqlRates rates[];
   ArraySetAsSeries(rates, false); // Du plus ancien au plus récent
   if(CopyRates(_Symbol, InpTimeFrame, 0, max_window + 1, rates) < max_window + 1)
   {
      //Print("Non rates");
      return;
   }
   // On utilise le prix de clôture ("close") des bougies 1H
   // On définit le décalage pour la régression (on utilise les 'periods' dernières valeurs)
   int offset = max_window - InpRegPeriods + 1; 
   double x_mean = (InpRegPeriods - 1) / 2.0;
   double y_mean = 0;
   
   for(int i = 0; i < InpRegPeriods; i++)
     {
      y_mean += rates[i + offset].close;
     }
   y_mean /= InpRegPeriods;

   double ss_xy = 0;
   double ss_x = 0;

   for(int i = 0; i < InpRegPeriods; i++)
     {
      double x_diff = i - x_mean;
      double y_diff = rates[i + offset].close - y_mean;
      ss_xy += x_diff * y_diff;
      ss_x += x_diff * x_diff;
     }

   double pente = 0;
   if(ss_x != 0)
      pente = ss_xy / ss_x;

   // Calcul du R2 sur les 'periods' dernières valeurs
   double ss_res = 0;
   double ss_tot = 0;
   for(int i = 0; i < InpRegPeriods; i++)
     {
      double x_diff = i - x_mean;
      double y_diff = rates[i + offset].close - y_mean;
      double y_hat = pente * x_diff + y_mean;
      
      ss_res += MathPow(rates[i + offset].close - y_hat, 2);
      ss_tot += MathPow(y_diff, 2);
     }
   
   double r2 = 0;
   if(ss_tot != 0)
      r2 = 1.0 - (ss_res / ss_tot);

   // Calcul de l'Efficiency Ratio (ER) sur InpErWindow périodes (dynamique et aligné)
   double change = MathAbs(rates[max_window].close - rates[max_window - InpErWindow].close);
   double vol_abs = 0;
   for(int i = 0; i < InpErWindow; i++)
     {
      vol_abs += MathAbs(rates[max_window - i].close - rates[max_window - i - 1].close);
     }
   double er = 0;
   if(vol_abs != 0)
      er = change / vol_abs;

   // y = mx + b
   double b = y_mean - pente * x_mean;

   // Point 1 (Début du segment de régression)
   double price1 = b;
   datetime time1 = rates[offset].time;

   // Point 2 (La bougie 1H actuelle)
   double price2 = pente * (InpRegPeriods - 1) + b;
   datetime time2 = rates[max_window].time;

   // Dessine la droite plus fine (épaisseur 1 au lieu de 2)
   string line_name = "Reg_Line_" + IntegerToString(InpRegPeriods);
   TrendCreate(0, line_name, 0, time1, price1, time2, price2, clrYellow, STYLE_SOLID, 1, false, false, false, true);
   
   // Affiche le texte du R2 et de l'ER au-dessus de la fin de la droite
   string text_name = "Reg_R2_Text";
   if(ObjectFind(0, text_name) < 0)
     {
      ObjectCreate(0, text_name, OBJ_TEXT, 0, time2, price2);
     }
   else
     {
      ObjectMove(0, text_name, 0, time2, price2);
     }
   ObjectSetString(0, text_name, OBJPROP_TEXT, "  R2: " + DoubleToString(r2, 2) + " | ER: " + DoubleToString(er, 2));
   ObjectSetInteger(0, text_name, OBJPROP_COLOR, clrYellow);
   ObjectSetInteger(0, text_name, OBJPROP_FONTSIZE, 10);
  }

//+------------------------------------------------------------------+
//| TradeTransaction function                                        |
//+------------------------------------------------------------------+
void OnTradeTransaction(const MqlTradeTransaction &trans,
                        const MqlTradeRequest &request,
                        const MqlTradeResult &result)
  {
//---
  }
//+----------------------------------------------------------------------------------------------------+
//| Crée une ligne de tendance aux coordonnées données                                                 |
//+----------------------------------------------------------------------------------------------------+
bool TrendCreate(const long            chart_ID=0,        // identifiant du graphique
                 const string          name="TrendLine",  // nom de la ligne
                 const int             sub_window=0,      // indice de sous-fenêtre
                 datetime              time1=0,           // heure du premier point
                 double                price1=0,          // prix du premier point
                 datetime              time2=0,           // heure du deuxième point
                 double                price2=0,          // prix du deuxième point
                 const color           clr=clrRed,        // couleur de la ligne
                 const ENUM_LINE_STYLE style=STYLE_SOLID, // style de la ligne
                 const int             width=1,           // largeur de la ligne
                 const bool            back=false,        // en arrière plan
                 const bool            selection=false,    // mise en surbrillance pour le déplacement
                 const bool            ray_left=false,    // Prolongement de la ligne vers la gauche
                 const bool            ray_right=false,   // Prolongement de la ligne vers la droite
                 const bool            hidden=false,       // caché dans la liste des objets
                 const long            z_order=0)         // priorité pour le clic de souris
  {
   ChangeTrendEmptyPoints(time1,price1,time2,price2);
   ResetLastError();
   if(!ObjectCreate(chart_ID,name,OBJ_TREND,sub_window,time1,price1,time2,price2))
     {
      // Si l'objet existe déjà, on le déplace simplement au lieu de spammer d'erreurs
      if (GetLastError() == 4200) {
          ObjectMove(chart_ID, name, 0, time1, price1);
          ObjectMove(chart_ID, name, 1, time2, price2);
          return true;
      }
      return(false);
     }
   ObjectSetInteger(chart_ID,name,OBJPROP_COLOR,clr);
   ObjectSetInteger(chart_ID,name,OBJPROP_STYLE,style);
   ObjectSetInteger(chart_ID,name,OBJPROP_WIDTH,width);
   ObjectSetInteger(chart_ID,name,OBJPROP_BACK,back);
   ObjectSetInteger(chart_ID,name,OBJPROP_SELECTABLE,selection);
   ObjectSetInteger(chart_ID,name,OBJPROP_SELECTED,selection);
   ObjectSetInteger(chart_ID,name,OBJPROP_RAY_LEFT,ray_left);
   ObjectSetInteger(chart_ID,name,OBJPROP_RAY_RIGHT,ray_right);
   ObjectSetInteger(chart_ID,name,OBJPROP_HIDDEN,hidden);
   ObjectSetInteger(chart_ID,name,OBJPROP_ZORDER,z_order);
   return(true);
  }
//+----------------------------------------------------------------------------------------------------+
//| Vérifie les valeurs des points d'ancrage de la ligne de tendance et définit la valeur              |
//| par défaut des points vides                                                                        |
//+----------------------------------------------------------------------------------------------------+
void ChangeTrendEmptyPoints(datetime &time1,double &price1,
                            datetime &time2,double &price2)
  {
   if(!time1) time1=TimeCurrent();
   if(!price1) price1=SymbolInfoDouble(Symbol(),SYMBOL_BID);
   if(!time2)
     {
      datetime temp[10];
      CopyTime(Symbol(),Period(),time1,10,temp);
      time2=temp[0];
     }
   if(!price2) price2=price1;
  }
//+----------------------------------------------------------------------------------------------------+
void ChangeEmptyPoint(datetime &time1,double &price1)
  {
   if(!time1) time1=TimeCurrent();
   if(!price1) price1=SymbolInfoDouble(Symbol(),SYMBOL_BID);
  }
//+------------------------------------------------------------------+
bool HLineCreate(const long            chart_ID=0,
                 const string          name="HLine",
                 const int             sub_window=0,
                 double                price=0,
                 const color           clr=clrRed,
                 const ENUM_LINE_STYLE style=STYLE_SOLID,
                 const int             width=1,
                 const bool            back=false,
                 const bool            selection=true,
                 const bool            hidden=true,
                 const long            z_order=0)
  {
   if(!price) price=SymbolInfoDouble(Symbol(),SYMBOL_BID);
   ResetLastError();
   if(!ObjectCreate(chart_ID,name,OBJ_HLINE,sub_window,0,price)) return(false);
   ObjectSetInteger(chart_ID,name,OBJPROP_COLOR,clr);
   ObjectSetInteger(chart_ID,name,OBJPROP_STYLE,style);
   ObjectSetInteger(chart_ID,name,OBJPROP_WIDTH,width);
   return(true);
  }
//+------------------------------------------------------------------+
bool HLineMove(const long   chart_ID=0,
               const string name="HLine",
               double       price=0)
  {
   if(!price) price=SymbolInfoDouble(Symbol(),SYMBOL_BID);
   ResetLastError();
   if(!ObjectMove(chart_ID,name,0,0,price)) return(false);
   return(true);
  }
//+------------------------------------------------------------------+
bool HLineDelete(const long   chart_ID=0,
                 const string name="HLine")
  {
   ResetLastError();
   if(!ObjectDelete(chart_ID,name)) return(false);
   return(true);
  }
//+------------------------------------------------------------------+
bool RectangleCreate(const long chart_ID=0, const string objectName="Rectangle", const int sub_window=0,
                     const int x=10, const int xsize=10, const int y=10, const int ysize=15, const color clr=Lime)
  {
   if(!ObjectCreate(0, objectName, OBJ_RECTANGLE_LABEL, 0, 0, 0)) return(false);
   ObjectSetInteger(0, objectName, OBJPROP_CORNER, CORNER_LEFT_UPPER);
   ObjectSetInteger(0, objectName, OBJPROP_XDISTANCE, x);
   ObjectSetInteger(0, objectName, OBJPROP_XSIZE, xsize);
   ObjectSetInteger(0, objectName, OBJPROP_BORDER_TYPE, BORDER_FLAT);
   ObjectSetInteger(0, objectName, OBJPROP_YDISTANCE, y);
   ObjectSetInteger(0, objectName, OBJPROP_YSIZE, ysize);
   ObjectSetInteger(0, objectName, OBJPROP_BGCOLOR, clr);
   ObjectSetInteger(0, objectName, OBJPROP_COLOR, clr);
   return(true);
  }
