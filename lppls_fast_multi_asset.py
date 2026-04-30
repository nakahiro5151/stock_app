"""
高速LPPLS信頼度指標システム - 複数資産対応版
処理時間を大幅短縮し、日経225、ドル円、金などの分析に対応
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
import yfinance as yf
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# 安全なフォント設定
def setup_safe_fonts():
    """安全なフォント設定"""
    try:
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['font.size'] = 10
        plt.rcParams['figure.figsize'] = (16, 12)
        plt.rcParams['axes.unicode_minus'] = False
        print("✅ Font setup completed")
    except Exception as e:
        print(f"⚠️ Font setup warning: {e}")

setup_safe_fonts()

class FastLPPLSAnalyzer:
    """高速LPPLS分析クラス"""
    
    # 資産クラス定義
    ASSET_SYMBOLS = {
        'SP500': '^GSPC',
        'Nikkei225': '^N225', 
        'USDJPY': 'USDJPY=X',
        'Gold': 'GC=F',
        'Bitcoin': 'BTC-USD',
        'NASDAQ': '^IXIC',
        'DAX': '^GDAXI',
        'FTSE': '^FTSE'
    }
    
    def __init__(self, asset_name="SP500", period="5y"):
        """
        初期化
        
        Args:
            asset_name (str): 資産名 ('SP500', 'Nikkei225', 'USDJPY', 'Gold', etc.)
            period (str): データ取得期間 (デフォルト: 5y)
        """
        self.asset_name = asset_name
        self.symbol = self.ASSET_SYMBOLS.get(asset_name, asset_name)
        self.period = period
        self.data = None
        self.confidence_series = []
        
    def fetch_data(self):
        """データ取得"""
        print(f"📈 Fetching {self.asset_name} ({self.symbol}) data for {self.period}...")
        
        try:
            ticker = yf.Ticker(self.symbol)
            self.data = ticker.history(period=self.period)
            
            if self.data.empty:
                raise ValueError(f"No data available for {self.symbol}")
                
            print(f"✅ Data acquired: {len(self.data)} days")
            return True
            
        except Exception as e:
            print(f"❌ Data acquisition error: {e}")
            return False
    
    def simple_lppls_fit(self, t, log_prices, tc_days_ahead=30):
        """
        簡略化されたLPPLSフィッティング（高速版）
        
        Args:
            t (array): 時間配列
            log_prices (array): ログ価格
            tc_days_ahead (int): 臨界時間（現在から何日後）
            
        Returns:
            tuple: (r_squared, confidence_score, bubble_risk)
        """
        try:
            tc = t[-1] + tc_days_ahead
            dt = tc - t
            dt = np.maximum(dt, 1e-10)
            
            # 簡略化されたLPPLS方程式のフィッティング
            # ln(p) = A + B*(tc-t)^m + C*(tc-t)^m*cos(w*ln(tc-t))
            
            # 固定パラメータで高速計算
            m = 0.3  # 固定値
            omega = 6.0  # 固定値
            
            # 線形回帰用の特徴量作成
            X = np.column_stack([
                np.ones(len(t)),  # A項
                dt ** m,  # B項
                (dt ** m) * np.cos(omega * np.log(dt))  # C項
            ])
            
            # 最小二乗法で係数を求める
            coeffs = np.linalg.lstsq(X, log_prices, rcond=None)[0]
            A, B, C = coeffs
            
            # 予測値の計算
            predicted = X @ coeffs
            
            # R²の計算
            ss_res = np.sum((log_prices - predicted) ** 2)
            ss_tot = np.sum((log_prices - np.mean(log_prices)) ** 2)
            r_squared = max(0, 1 - (ss_res / ss_tot))
            
            # 信頼度スコアの計算（簡略版）
            confidence_score = self._fast_confidence_score(r_squared, B, C, m, omega)
            
            # バブルリスクの計算（簡略版）
            bubble_risk = self._fast_bubble_risk(r_squared, B, C, tc_days_ahead)
            
            return r_squared, confidence_score, bubble_risk
            
        except Exception as e:
            return 0, 0, 0
    
    def _fast_confidence_score(self, r_squared, B, C, m, omega):
        """高速信頼度スコア計算"""
        score = 0
        
        # R²による評価 (0-50点)
        score += min(r_squared * 60, 50)
        
        # パラメータ評価 (0-50点)
        if B < 0:  # Bは負であるべき
            score += 15
        
        if abs(C) > 0.01:  # 振動成分の存在
            score += 20
        elif abs(C) > 0.005:
            score += 10
        
        if 0.1 < m < 0.9:  # m値の妥当性
            score += 10
        
        if 2 < omega < 15:  # omega値の妥当性
            score += 5
        
        return min(score, 100)
    
    def _fast_bubble_risk(self, r_squared, B, C, tc_days_ahead):
        """高速バブルリスク計算"""
        risk = 0
        
        # 高いR²値
        if r_squared > 0.6:
            risk += 30
        elif r_squared > 0.4:
            risk += 15
        
        # 短期間での臨界点
        if tc_days_ahead < 30:
            risk += 40
        elif tc_days_ahead < 60:
            risk += 25
        elif tc_days_ahead < 90:
            risk += 15
        
        # 強い振動成分
        if abs(C) > 0.05:
            risk += 20
        elif abs(C) > 0.02:
            risk += 10
        
        # 負のB値（パワーロー成分）
        if B < -0.05:
            risk += 10
        
        return min(risk, 100)
    
    def calculate_fast_confidence_series(self, window_size=126, step_size=42):
        """
        高速信頼度指標計算（解像度を下げて高速化）
        
        Args:
            window_size (int): ウィンドウサイズ（デフォルト: 126日 = 約6ヶ月）
            step_size (int): ステップサイズ（デフォルト: 42日 = 約6週間）
        """
        if self.data is None:
            return None
            
        prices = self.data['Close'].values
        log_prices = np.log(prices)
        dates = self.data.index
        t = np.arange(len(prices))
        
        print(f"🔍 Calculating fast confidence series...")
        print(f"   Window size: {window_size} days (~{window_size//21:.1f} months)")
        print(f"   Step size: {step_size} days (~{step_size//7:.1f} weeks)")
        
        confidence_data = []
        
        # 複数の臨界時間候補で分析（高速化のため3つのみ）
        tc_candidates = [30, 60, 90]  # 1ヶ月、2ヶ月、3ヶ月後
        
        # スライディングウィンドウで分析
        for i in range(window_size, len(prices), step_size):
            start_idx = i - window_size
            end_idx = i
            
            t_window = t[start_idx:end_idx]
            log_prices_window = log_prices[start_idx:end_idx]
            date_window = dates[end_idx-1]
            
            # 複数の臨界時間で分析し、最良の結果を選択
            best_r_squared = 0
            best_confidence = 0
            best_bubble_risk = 0
            
            for tc_days in tc_candidates:
                r_squared, confidence, bubble_risk = self.simple_lppls_fit(
                    t_window, log_prices_window, tc_days
                )
                
                if r_squared > best_r_squared:
                    best_r_squared = r_squared
                    best_confidence = confidence
                    best_bubble_risk = bubble_risk
            
            confidence_data.append({
                'date': date_window,
                'price': prices[end_idx-1],
                'confidence': best_confidence,
                'bubble_risk': best_bubble_risk,
                'r_squared': best_r_squared
            })
            
            # 進捗表示
            if len(confidence_data) % 5 == 0:
                print(f"   Processed: {len(confidence_data)} windows")
        
        self.confidence_series = pd.DataFrame(confidence_data)
        print(f"✅ Fast confidence series calculated: {len(self.confidence_series)} data points")
        
        return self.confidence_series
    
    def plot_fast_analysis(self):
        """高速分析結果のプロット - 2軸表示"""
        if self.confidence_series is None or len(self.confidence_series) == 0:
            print("❌ No confidence data available")
            return
        
        fig, axes = plt.subplots(2, 1, figsize=(16, 12))
        fig.suptitle(f'{self.asset_name} - LPPLS Dual-Axis Analysis', 
                     fontsize=16, fontweight='bold')
        
        dates = self.confidence_series['date']
        prices = self.confidence_series['price']
        confidence = self.confidence_series['confidence']
        bubble_risk = self.confidence_series['bubble_risk']
        
        # 1. 株価 + 信頼度指標（2軸表示）
        ax1 = axes[0]
        
        # 左軸: 株価（青線）
        color_price1 = 'tab:blue'
        ax1.set_xlabel('Date', fontsize=11)
        ax1.set_ylabel(f'{self.asset_name} Price', color=color_price1, fontsize=12, fontweight='bold')
        line1 = ax1.plot(dates, prices, color=color_price1, linewidth=2.5, label=f'{self.asset_name} Price')
        ax1.tick_params(axis='y', labelcolor=color_price1, labelsize=10)
        
        # 資産に応じてログスケールを適用
        if self.asset_name in ['SP500', 'Nikkei225', 'NASDAQ', 'Gold', 'Bitcoin']:
            ax1.set_yscale('log')
            ax1.set_ylabel(f'{self.asset_name} Price (Log Scale)', color=color_price1, fontsize=12, fontweight='bold')
        
        ax1.grid(True, alpha=0.3)
        
        # 右軸: 信頼度指標（緑線）
        ax1_twin = ax1.twinx()
        color_conf = 'tab:green'
        ax1_twin.set_ylabel('LPPLS Confidence (%)', color=color_conf, fontsize=12, fontweight='bold')
        line2 = ax1_twin.plot(dates, confidence, color=color_conf, linewidth=2.5, label='LPPLS Confidence')
        ax1_twin.fill_between(dates, 0, confidence, alpha=0.25, color=color_conf)
        ax1_twin.tick_params(axis='y', labelcolor=color_conf, labelsize=10)
        ax1_twin.set_ylim(0, 100)
        
        # 信頼度の閾値線
        ax1_twin.axhline(y=50, color='orange', linestyle='--', alpha=0.7, linewidth=1.5, label='Moderate (50%)')
        ax1_twin.axhline(y=70, color='darkred', linestyle='--', alpha=0.7, linewidth=1.5, label='High (70%)')
        
        # 統合凡例
        lines1 = line1 + line2
        labels1 = [l.get_label() for l in lines1] + ['Moderate (50%)', 'High (70%)']
        ax1.legend(lines1 + [plt.Line2D([0], [0], color='orange', linestyle='--'), 
                            plt.Line2D([0], [0], color='darkred', linestyle='--')], 
                  labels1, loc='upper left', fontsize=10)
        
        ax1.set_title(f'Price vs LPPLS Confidence - Dual Axis View', fontsize=13, fontweight='bold')
        
        # 2. 株価 + バブルリスク指標（2軸表示）
        ax2 = axes[1]
        
        # 左軸: 株価（青線）
        color_price2 = 'tab:blue'
        ax2.set_xlabel('Date', fontsize=11)
        ax2.set_ylabel(f'{self.asset_name} Price', color=color_price2, fontsize=12, fontweight='bold')
        line3 = ax2.plot(dates, prices, color=color_price2, linewidth=2.5, label=f'{self.asset_name} Price')
        ax2.tick_params(axis='y', labelcolor=color_price2, labelsize=10)
        
        # 資産に応じてログスケールを適用
        if self.asset_name in ['SP500', 'Nikkei225', 'NASDAQ', 'Gold', 'Bitcoin']:
            ax2.set_yscale('log')
            ax2.set_ylabel(f'{self.asset_name} Price (Log Scale)', color=color_price2, fontsize=12, fontweight='bold')
        
        ax2.grid(True, alpha=0.3)
        
        # 右軸: バブルリスク指標（赤線）
        ax2_twin = ax2.twinx()
        color_risk = 'tab:red'
        ax2_twin.set_ylabel('Bubble Risk (%)', color=color_risk, fontsize=12, fontweight='bold')
        line4 = ax2_twin.plot(dates, bubble_risk, color=color_risk, linewidth=2.5, label='Bubble Risk')
        ax2_twin.fill_between(dates, 0, bubble_risk, alpha=0.25, color=color_risk)
        ax2_twin.tick_params(axis='y', labelcolor=color_risk, labelsize=10)
        ax2_twin.set_ylim(0, 100)
        
        # バブルリスクの閾値線
        ax2_twin.axhline(y=30, color='orange', linestyle='--', alpha=0.7, linewidth=1.5, label='Warning (30%)')
        ax2_twin.axhline(y=60, color='darkred', linestyle='--', alpha=0.7, linewidth=1.5, label='Critical (60%)')
        
        # 統合凡例
        lines2 = line3 + line4
        labels2 = [l.get_label() for l in lines2] + ['Warning (30%)', 'Critical (60%)']
        ax2.legend(lines2 + [plt.Line2D([0], [0], color='orange', linestyle='--'), 
                            plt.Line2D([0], [0], color='darkred', linestyle='--')], 
                  labels2, loc='upper left', fontsize=10)
        
        ax2.set_title(f'Price vs Bubble Risk - Dual Axis View', fontsize=13, fontweight='bold')
        
        # 日付フォーマット設定
        for ax in [ax1, ax2]:
            ax.xaxis.set_major_formatter(DateFormatter('%Y'))
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=6))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha='center', fontsize=10)
        
        plt.tight_layout()
        plt.show()
        
        # 2軸表示の説明
        print("\n📊 DUAL-AXIS CHART EXPLANATION:")
        print("="*50)
        print("Upper Chart: Price (Left Axis) + LPPLS Confidence (Right Axis)")
        print("  • Blue Line: Asset Price")
        print("  • Green Line: LPPLS Model Confidence (0-100%)")
        print("  • Green Area: Confidence Level Visualization")
        print()
        print("Lower Chart: Price (Left Axis) + Bubble Risk (Right Axis)")
        print("  • Blue Line: Asset Price")
        print("  • Red Line: Bubble Risk Indicator (0-100%)")
        print("  • Red Area: Risk Level Visualization")
        print("="*50)
        
        # 統計サマリー
        self._display_fast_summary()
    
    def _display_fast_summary(self):
        """高速分析サマリーの表示"""
        if len(self.confidence_series) == 0:
            return
            
        confidence = self.confidence_series['confidence']
        bubble_risk = self.confidence_series['bubble_risk']
        
        print(f"\n📊 {self.asset_name} - FAST LPPLS ANALYSIS SUMMARY")
        print("="*60)
        print(f"Analysis Period: {self.confidence_series['date'].iloc[0].date()} to {self.confidence_series['date'].iloc[-1].date()}")
        print(f"Data Points: {len(self.confidence_series)}")
        print()
        print(f"Average Confidence: {confidence.mean():.1f}%")
        print(f"Average Bubble Risk: {bubble_risk.mean():.1f}%")
        print(f"High Risk Periods (>60%): {(bubble_risk > 60).sum()} ({(bubble_risk > 60).mean()*100:.1f}%)")
        print()
        
        # 最新状態
        latest = self.confidence_series.iloc[-1]
        print(f"🔍 CURRENT STATUS:")
        print(f"  Confidence: {latest['confidence']:.1f}%")
        print(f"  Bubble Risk: {latest['bubble_risk']:.1f}%")
        
        if latest['bubble_risk'] > 60:
            print("  🔴 HIGH BUBBLE RISK")
        elif latest['bubble_risk'] > 30:
            print("  🟠 MODERATE BUBBLE RISK")
        else:
            print("  🟢 LOW BUBBLE RISK")
        print("="*60)
    
    def run_fast_analysis(self, window_size=126, step_size=42):
        """高速分析を実行"""
        print(f"🚀 Starting Fast LPPLS Analysis for {self.asset_name}")
        print("="*60)
        
        # データ取得
        if not self.fetch_data():
            return False
        
        # 高速信頼度指標の計算
        confidence_data = self.calculate_fast_confidence_series(window_size, step_size)
        
        if confidence_data is not None and len(confidence_data) > 0:
            # 結果をプロット
            self.plot_fast_analysis()
            return True
        else:
            print("❌ Analysis failed")
            return False

class MultiAssetLPPLSAnalyzer:
    """複数資産の一括分析クラス"""
    
    def __init__(self, assets=None, period="3y"):
        """
        初期化
        
        Args:
            assets (list): 分析する資産リスト
            period (str): データ取得期間
        """
        if assets is None:
            assets = ['SP500', 'Nikkei225', 'USDJPY', 'Gold']
        
        self.assets = assets
        self.period = period
        self.analyzers = {}
        self.results = {}
    
    def run_multi_asset_analysis(self, window_size=126, step_size=42):
        """複数資産の一括分析"""
        print("🌍 Starting Multi-Asset LPPLS Analysis")
        print("="*60)
        
        for asset in self.assets:
            print(f"\n📊 Analyzing {asset}...")
            
            try:
                analyzer = FastLPPLSAnalyzer(asset, self.period)
                success = analyzer.run_fast_analysis(window_size, step_size)
                
                if success:
                    self.analyzers[asset] = analyzer
                    self.results[asset] = analyzer.confidence_series
                    print(f"✅ {asset} analysis completed")
                else:
                    print(f"❌ {asset} analysis failed")
                    
            except Exception as e:
                print(f"❌ {asset} analysis error: {e}")
        
        # 比較サマリーの表示
        self._display_comparison_summary()
        
        return len(self.results) > 0
    
    def _display_comparison_summary(self):
        """比較サマリーの表示"""
        if not self.results:
            return
            
        print("\n" + "="*80)
        print("🌍 MULTI-ASSET COMPARISON SUMMARY")
        print("="*80)
        
        comparison_data = []
        
        for asset, data in self.results.items():
            if len(data) > 0:
                latest = data.iloc[-1]
                avg_confidence = data['confidence'].mean()
                avg_bubble_risk = data['bubble_risk'].mean()
                high_risk_periods = (data['bubble_risk'] > 60).mean() * 100
                
                comparison_data.append({
                    'Asset': asset,
                    'Current Confidence': f"{latest['confidence']:.1f}%",
                    'Current Bubble Risk': f"{latest['bubble_risk']:.1f}%",
                    'Avg Confidence': f"{avg_confidence:.1f}%",
                    'Avg Bubble Risk': f"{avg_bubble_risk:.1f}%",
                    'High Risk %': f"{high_risk_periods:.1f}%"
                })
        
        # 表形式で表示
        if comparison_data:
            df = pd.DataFrame(comparison_data)
            print(df.to_string(index=False))
            
            # 最もリスクの高い資産
            current_risks = [float(item['Current Bubble Risk'].rstrip('%')) for item in comparison_data]
            max_risk_idx = np.argmax(current_risks)
            highest_risk_asset = comparison_data[max_risk_idx]['Asset']
            highest_risk_value = current_risks[max_risk_idx]
            
            print(f"\n🔴 HIGHEST CURRENT RISK: {highest_risk_asset} ({highest_risk_value:.1f}%)")
        
        print("="*80)

# 使用例とデモ
def demo_fast_single_asset(asset_name="SP500"):
    """単一資産の高速分析デモ"""
    print(f"🚀 Fast LPPLS Analysis Demo - {asset_name}")
    print("="*60)
    
    analyzer = FastLPPLSAnalyzer(asset_name, period="3y")
    
    if analyzer.run_fast_analysis(window_size=126, step_size=42):
        return analyzer
    else:
        print("❌ Demo analysis failed")
        return None

def demo_multi_asset_analysis():
    """複数資産の一括分析デモ"""
    print("🌍 Multi-Asset LPPLS Analysis Demo")
    print("="*60)
    
    # 分析する資産リスト
    assets = ['SP500', 'Nikkei225', 'USDJPY', 'Gold', 'Bitcoin']
    
    analyzer = MultiAssetLPPLSAnalyzer(assets, period="3y")
    
    if analyzer.run_multi_asset_analysis(window_size=126, step_size=42):
        return analyzer
    else:
        print("❌ Multi-asset analysis failed")
        return None

if __name__ == "__main__":
    # 利用可能な資産の表示
    print("📋 Available Assets:")
    for name, symbol in FastLPPLSAnalyzer.ASSET_SYMBOLS.items():
        print(f"  {name}: {symbol}")
    print()
    
    # デモ実行の選択
    print("Choose analysis type:")
    print("1. Single Asset (fast)")
    print("2. Multi-Asset (comprehensive)")
    
    # デフォルトで複数資産分析を実行
    demo_result = demo_multi_asset_analysis()