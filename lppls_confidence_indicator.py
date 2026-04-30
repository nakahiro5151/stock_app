"""
LPPLS信頼度指標システム - ドットコムバブル風の出力
時系列での信頼度指標とバブル検出シグナルを表示
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
import yfinance as yf
from scipy.optimize import differential_evolution
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

class LPPLSConfidenceIndicator:
    """LPPLS信頼度指標クラス"""
    
    def __init__(self, symbol="^GSPC", period="10y"):
        self.symbol = symbol
        self.period = period
        self.data = None
        self.confidence_series = []
        self.bubble_signals = []
        
    def fetch_data(self):
        """データ取得"""
        print(f"📈 Fetching {self.symbol} data for {self.period}...")
        
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
    
    def lppls_equation(self, t, params, tc):
        """LPPLS方程式"""
        A, B, C, m, omega, phi = params
        dt = tc - t
        dt = np.maximum(dt, 1e-10)
        log_price = A + B * (dt ** m) + C * (dt ** m) * np.cos(omega * np.log(dt) + phi)
        return log_price
    
    def objective_function(self, params, t, log_prices, tc):
        """最適化目的関数"""
        try:
            predicted = self.lppls_equation(t, params, tc)
            residuals = log_prices - predicted
            return np.sum(residuals ** 2)
        except:
            return 1e10
    
    def fit_single_window(self, t_window, log_prices_window, tc_range_days=60):
        """単一ウィンドウでのLPPLSフィッティング"""
        best_r_squared = -1
        best_params = None
        best_tc = None
        
        # 臨界時間の探索範囲
        tc_min = t_window[-1] + 5
        tc_max = t_window[-1] + tc_range_days
        tc_candidates = np.linspace(tc_min, tc_max, 10)
        
        for tc in tc_candidates:
            # パラメータの制約
            bounds = [
                (log_prices_window.min() - 0.5, log_prices_window.max() + 0.5),  # A
                (-1, 0),                                                          # B
                (-0.2, 0.2),                                                     # C
                (0.1, 0.9),                                                      # m
                (2, 15),                                                         # omega
                (-np.pi, np.pi)                                                  # phi
            ]
            
            try:
                result = differential_evolution(
                    self.objective_function,
                    bounds,
                    args=(t_window, log_prices_window, tc),
                    seed=42,
                    maxiter=100,
                    popsize=10
                )
                
                if result.success:
                    predicted = self.lppls_equation(t_window, result.x, tc)
                    ss_res = np.sum((log_prices_window - predicted) ** 2)
                    ss_tot = np.sum((log_prices_window - np.mean(log_prices_window)) ** 2)
                    r_squared = 1 - (ss_res / ss_tot)
                    
                    if r_squared > best_r_squared:
                        best_r_squared = r_squared
                        best_params = result.x
                        best_tc = tc
                        
            except:
                continue
        
        return best_r_squared, best_params, best_tc
    
    def calculate_confidence_series(self, window_size=252, step_size=21):
        """時系列での信頼度指標を計算"""
        if self.data is None:
            return None
            
        prices = self.data['Close'].values
        log_prices = np.log(prices)
        dates = self.data.index
        t = np.arange(len(prices))
        
        print(f"🔍 Calculating confidence series...")
        print(f"   Window size: {window_size} days")
        print(f"   Step size: {step_size} days")
        
        confidence_data = []
        
        # スライディングウィンドウで分析
        for i in range(window_size, len(prices), step_size):
            start_idx = i - window_size
            end_idx = i
            
            t_window = t[start_idx:end_idx]
            log_prices_window = log_prices[start_idx:end_idx]
            date_window = dates[end_idx-1]
            
            # LPPLSフィッティング
            r_squared, params, tc = self.fit_single_window(t_window, log_prices_window)
            
            if r_squared > 0:
                # 信頼度スコアの計算
                confidence_score = self._calculate_confidence_score(r_squared, params, tc, t_window[-1])
                
                # バブル指標の計算
                bubble_indicator = self._calculate_bubble_indicator(r_squared, params, tc, t_window[-1])
                
                confidence_data.append({
                    'date': date_window,
                    'price': prices[end_idx-1],
                    'confidence': confidence_score,
                    'bubble_indicator': bubble_indicator,
                    'r_squared': r_squared,
                    'days_to_critical': tc - t_window[-1] if tc else np.nan,
                    'params': params
                })
            
            # 進捗表示
            if len(confidence_data) % 10 == 0:
                print(f"   Processed: {len(confidence_data)} windows")
        
        self.confidence_series = pd.DataFrame(confidence_data)
        print(f"✅ Confidence series calculated: {len(self.confidence_series)} data points")
        
        return self.confidence_series
    
    def _calculate_confidence_score(self, r_squared, params, tc, current_t):
        """信頼度スコアの計算"""
        if params is None:
            return 0
            
        A, B, C, m, omega, phi = params
        score = 0
        
        # R²による評価 (0-40点)
        score += min(r_squared * 50, 40)
        
        # パラメータの妥当性評価 (0-60点)
        # m値の評価
        if 0.1 < m < 0.9:
            score += 15
        elif 0.05 < m < 0.95:
            score += 10
        
        # omega値の評価
        if 2 < omega < 15:
            score += 15
        elif 1 < omega < 20:
            score += 10
        
        # C値の評価（振動の存在）
        if abs(C) > 0.01:
            score += 15
        elif abs(C) > 0.005:
            score += 10
        
        # 臨界時間の妥当性
        if tc and 5 < (tc - current_t) < 120:
            score += 15
        elif tc and 1 < (tc - current_t) < 250:
            score += 10
        
        return min(score, 100)
    
    def _calculate_bubble_indicator(self, r_squared, params, tc, current_t):
        """バブル指標の計算"""
        if params is None or tc is None:
            return 0
            
        A, B, C, m, omega, phi = params
        days_to_critical = tc - current_t
        
        bubble_score = 0
        
        # 高いR²値
        if r_squared > 0.7:
            bubble_score += 30
        elif r_squared > 0.5:
            bubble_score += 20
        
        # 短期間での臨界点
        if days_to_critical < 30:
            bubble_score += 40
        elif days_to_critical < 60:
            bubble_score += 25
        elif days_to_critical < 120:
            bubble_score += 15
        
        # 強い振動成分
        if abs(C) > 0.05:
            bubble_score += 20
        elif abs(C) > 0.02:
            bubble_score += 10
        
        # 適切なm値
        if 0.2 < m < 0.8:
            bubble_score += 10
        
        return min(bubble_score, 100)
    
    def plot_confidence_indicator(self):
        """信頼度指標のプロット（ドットコムバブル風）"""
        if self.confidence_series is None or len(self.confidence_series) == 0:
            print("❌ No confidence data available")
            return
        
        fig, axes = plt.subplots(4, 1, figsize=(16, 14))
        fig.suptitle(f'{self.symbol} - LPPLS Confidence Indicator Analysis', 
                     fontsize=16, fontweight='bold')
        
        dates = self.confidence_series['date']
        prices = self.confidence_series['price']
        confidence = self.confidence_series['confidence']
        bubble_indicator = self.confidence_series['bubble_indicator']
        
        # 1. 価格チャート（ログスケール）
        ax1 = axes[0]
        ax1.plot(dates, prices, 'b-', linewidth=1.5, label='S&P 500')
        ax1.set_ylabel('Price (Log Scale)', fontsize=11)
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_title('S&P 500 Index', fontsize=12)
        
        # 重要なイベントをマーク（例：ドットコムバブル、リーマンショック、コロナショック）
        self._add_historical_events(ax1)
        
        # 2. LPPLS信頼度指標
        ax2 = axes[1]
        ax2.fill_between(dates, 0, confidence, alpha=0.6, color='green', label='LPPLS Confidence')
        ax2.plot(dates, confidence, 'darkgreen', linewidth=1)
        ax2.axhline(y=50, color='orange', linestyle='--', alpha=0.7, label='Moderate Threshold')
        ax2.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='High Threshold')
        ax2.set_ylabel('Confidence (%)', fontsize=11)
        ax2.set_ylim(0, 100)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_title('LPPLS Model Confidence', fontsize=12)
        
        # 3. バブル指標
        ax3 = axes[2]
        # バブル指標を線グラフで表示
        ax3.plot(dates, bubble_indicator, 'red', linewidth=2, label='Bubble Risk')
        ax3.fill_between(dates, 0, bubble_indicator, alpha=0.3, color='red')
        ax3.axhline(y=30, color='orange', linestyle='--', alpha=0.7, label='Warning Level')
        ax3.axhline(y=60, color='red', linestyle='--', alpha=0.7, label='Critical Level')
        ax3.set_ylabel('Bubble Risk (%)', fontsize=11)
        ax3.set_ylim(0, 100)
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        ax3.set_title('Bubble Risk Indicator', fontsize=12)
        
        # 4. R²値の時系列
        ax4 = axes[3]
        r_squared_values = self.confidence_series['r_squared']
        ax4.plot(dates, r_squared_values, 'purple', linewidth=1.5, label='R² Value')
        ax4.fill_between(dates, 0, r_squared_values, alpha=0.3, color='purple')
        ax4.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='Good Fit')
        ax4.axhline(y=0.7, color='red', linestyle='--', alpha=0.7, label='Excellent Fit')
        ax4.set_ylabel('R² Value', fontsize=11)
        ax4.set_ylim(0, 1)
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        ax4.set_title('Model Fit Quality (R²)', fontsize=12)
        
        # 全軸に共通の日付フォーマット設定
        for ax in axes:
            ax.xaxis.set_major_formatter(DateFormatter('%Y'))
            ax.xaxis.set_major_locator(mdates.YearLocator(2))
            ax.xaxis.set_minor_locator(mdates.YearLocator())
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha='center')
        
        # 最下段のみX軸ラベル表示
        axes[-1].set_xlabel('Year', fontsize=11)
        
        plt.tight_layout()
        plt.show()
        
        # 統計サマリーの表示
        self._display_statistics_summary()
    
    def _add_historical_events(self, ax):
        """歴史的イベントをマーク"""
        events = [
            ('2000-03-10', 'Dot-com Peak', 'red'),
            ('2007-10-09', 'Pre-Financial Crisis', 'orange'),
            ('2020-02-19', 'Pre-COVID Crash', 'purple'),
        ]
        
        for date_str, label, color in events:
            try:
                event_date = pd.to_datetime(date_str)
                if self.data.index[0] <= event_date <= self.data.index[-1]:
                    ax.axvline(x=event_date, color=color, linestyle=':', alpha=0.8, linewidth=2)
                    ax.text(event_date, ax.get_ylim()[1] * 0.9, label, 
                           rotation=90, ha='right', va='top', color=color, fontsize=9)
            except:
                continue
    
    def _display_statistics_summary(self):
        """統計サマリーの表示"""
        if len(self.confidence_series) == 0:
            return
            
        confidence = self.confidence_series['confidence']
        bubble_indicator = self.confidence_series['bubble_indicator']
        r_squared = self.confidence_series['r_squared']
        
        print("\n" + "="*80)
        print("📊 LPPLS CONFIDENCE INDICATOR STATISTICS")
        print("="*80)
        print(f"Analysis Period: {self.confidence_series['date'].iloc[0].date()} to {self.confidence_series['date'].iloc[-1].date()}")
        print(f"Total Data Points: {len(self.confidence_series)}")
        print()
        print("CONFIDENCE STATISTICS:")
        print(f"  Average Confidence: {confidence.mean():.1f}%")
        print(f"  Maximum Confidence: {confidence.max():.1f}%")
        print(f"  High Confidence Periods (>70%): {(confidence > 70).sum()} ({(confidence > 70).mean()*100:.1f}%)")
        print()
        print("BUBBLE RISK STATISTICS:")
        print(f"  Average Bubble Risk: {bubble_indicator.mean():.1f}%")
        print(f"  Maximum Bubble Risk: {bubble_indicator.max():.1f}%")
        print(f"  Critical Risk Periods (>60%): {(bubble_indicator > 60).sum()} ({(bubble_indicator > 60).mean()*100:.1f}%)")
        print()
        print("MODEL FIT STATISTICS:")
        print(f"  Average R²: {r_squared.mean():.3f}")
        print(f"  Maximum R²: {r_squared.max():.3f}")
        print(f"  Good Fit Periods (R²>0.5): {(r_squared > 0.5).sum()} ({(r_squared > 0.5).mean()*100:.1f}%)")
        print("="*80)
        
        # 最新の状態
        latest = self.confidence_series.iloc[-1]
        print(f"\n🔍 CURRENT STATUS (as of {latest['date'].date()}):")
        print(f"  Current Confidence: {latest['confidence']:.1f}%")
        print(f"  Current Bubble Risk: {latest['bubble_indicator']:.1f}%")
        print(f"  Current R²: {latest['r_squared']:.3f}")
        
        if latest['bubble_indicator'] > 60:
            print("  🔴 STATUS: HIGH BUBBLE RISK")
        elif latest['bubble_indicator'] > 30:
            print("  🟠 STATUS: MODERATE BUBBLE RISK")
        else:
            print("  🟢 STATUS: LOW BUBBLE RISK")
    
    def run_full_analysis(self, window_size=252, step_size=21):
        """完全な分析を実行"""
        print("🐉 Starting LPPLS Confidence Indicator Analysis")
        print("="*60)
        
        # データ取得
        if not self.fetch_data():
            return False
        
        # 信頼度指標の計算
        confidence_data = self.calculate_confidence_series(window_size, step_size)
        
        if confidence_data is not None and len(confidence_data) > 0:
            # 結果をプロット
            self.plot_confidence_indicator()
            return True
        else:
            print("❌ Analysis failed")
            return False

# 使用例
def demo_confidence_indicator():
    """信頼度指標のデモ実行"""
    print("🚀 LPPLS Confidence Indicator Demo")
    print("="*60)
    
    # SP500の10年分データで分析
    analyzer = LPPLSConfidenceIndicator("^GSPC", period="10y")
    
    if analyzer.run_full_analysis(window_size=252, step_size=21):
        return analyzer
    else:
        print("❌ Demo analysis failed")
        return None

if __name__ == "__main__":
    # デモ実行
    demo_result = demo_confidence_indicator()