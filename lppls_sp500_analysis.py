"""
本格的なLPPLS (Log-Periodic Power Law Singularity) 実装
ソネット氏のドラゴンキング理論に基づくSP500分析システム

LPPLS方程式:
ln(p(t)) = A + B*(tc-t)^m + C*(tc-t)^m*cos(ω*ln(tc-t) + φ)

パラメータ:
- A: ログ価格の基準レベル
- B: パワーロー成分の振幅
- C: ログ周期振動の振幅  
- m: 臨界指数 (0 < m < 1)
- ω: ログ周期振動の角周波数
- φ: 位相
- tc: 臨界時間
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
import yfinance as yf
from scipy.optimize import minimize, differential_evolution
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Google Colab環境設定
try:
    from google.colab import drive
    drive.mount('/content/drive')
    IN_COLAB = True
except:
    IN_COLAB = False

# 安全なフォント設定
def setup_safe_fonts():
    """安全なフォント設定"""
    try:
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['font.size'] = 12
        plt.rcParams['figure.figsize'] = (16, 12)
        plt.rcParams['axes.unicode_minus'] = False
        print("✅ Font setup completed")
    except Exception as e:
        print(f"⚠️ Font setup warning: {e}")

setup_safe_fonts()

class LPPLSAnalyzer:
    """LPPLS (Log-Periodic Power Law Singularity) 分析クラス"""
    
    def __init__(self, symbol="^GSPC", period="10y"):
        """
        初期化
        
        Args:
            symbol (str): 株式シンボル (デフォルト: ^GSPC = S&P500)
            period (str): データ取得期間 (デフォルト: 10y = 10年)
        """
        self.symbol = symbol
        self.period = period
        self.data = None
        self.lppls_results = {}
        
    def fetch_data(self):
        """SP500データを取得"""
        print(f"📈 Fetching {self.symbol} data for {self.period}...")
        
        try:
            ticker = yf.Ticker(self.symbol)
            self.data = ticker.history(period=self.period)
            
            if self.data.empty:
                raise ValueError(f"No data available for {self.symbol}")
                
            print(f"✅ Data acquired: {len(self.data)} days")
            print(f"📅 Period: {self.data.index[0].date()} to {self.data.index[-1].date()}")
            return True
            
        except Exception as e:
            print(f"❌ Data acquisition error: {e}")
            return False
    
    def lppls_equation(self, t, params, tc):
        """
        LPPLS方程式
        
        Args:
            t (array): 時間配列
            params (list): [A, B, C, m, omega, phi]
            tc (float): 臨界時間
            
        Returns:
            array: 予測ログ価格
        """
        A, B, C, m, omega, phi = params
        
        # tc-tの計算（ゼロ除算を防ぐ）
        dt = tc - t
        dt = np.maximum(dt, 1e-10)
        
        # LPPLS方程式
        log_price = A + B * (dt ** m) + C * (dt ** m) * np.cos(omega * np.log(dt) + phi)
        
        return log_price
    
    def objective_function(self, params, t, log_prices, tc):
        """
        最適化目的関数（最小二乗法）
        
        Args:
            params (list): [A, B, C, m, omega, phi]
            t (array): 時間配列
            log_prices (array): 実際のログ価格
            tc (float): 臨界時間
            
        Returns:
            float: 残差二乗和
        """
        try:
            predicted = self.lppls_equation(t, params, tc)
            residuals = log_prices - predicted
            return np.sum(residuals ** 2)
        except:
            return 1e10
    
    def fit_lppls_model(self, window_size=500, tc_range_days=250):
        """
        LPPLS モデルのフィッティング
        
        Args:
            window_size (int): フィッティングウィンドウサイズ
            tc_range_days (int): 臨界時間の探索範囲（日数）
        """
        if self.data is None:
            print("❌ No data available. Please fetch data first.")
            return None
            
        prices = self.data['Close'].values
        log_prices = np.log(prices)
        dates = self.data.index
        
        # 時間を数値に変換（日数）
        t = np.arange(len(prices))
        
        # 最新のwindow_sizeデータを使用
        if len(prices) < window_size:
            window_size = len(prices)
            
        t_window = t[-window_size:]
        log_prices_window = log_prices[-window_size:]
        dates_window = dates[-window_size:]
        
        print(f"🔍 Fitting LPPLS model...")
        print(f"   Window size: {window_size} days")
        print(f"   Date range: {dates_window[0].date()} to {dates_window[-1].date()}")
        
        best_fit = None
        best_error = float('inf')
        best_tc = None
        best_r_squared = -1
        
        # 臨界時間の探索範囲
        tc_min = t_window[-1] + 10  # 最低10日後
        tc_max = t_window[-1] + tc_range_days
        
        # 複数の臨界時間で最適化を試行
        tc_candidates = np.linspace(tc_min, tc_max, 20)
        
        for tc in tc_candidates:
            # パラメータの初期値と制約
            # A: ログ価格の基準レベル
            A_init = log_prices_window[-1]
            
            # B: パワーロー成分（通常負の値）
            B_init = -0.1
            
            # C: ログ周期振動の振幅
            C_init = 0.01
            
            # m: 臨界指数 (0 < m < 1)
            m_init = 0.3
            
            # omega: ログ周期振動の角周波数
            omega_init = 6.0
            
            # phi: 位相
            phi_init = 0.0
            
            initial_params = [A_init, B_init, C_init, m_init, omega_init, phi_init]
            
            # パラメータの制約
            bounds = [
                (log_prices_window.min() - 1, log_prices_window.max() + 1),  # A
                (-2, 0),                                                      # B
                (-0.5, 0.5),                                                 # C
                (0.01, 0.99),                                                # m
                (1, 25),                                                     # omega
                (-2*np.pi, 2*np.pi)                                          # phi
            ]
            
            try:
                # 差分進化アルゴリズムで最適化
                result = differential_evolution(
                    self.objective_function,
                    bounds,
                    args=(t_window, log_prices_window, tc),
                    seed=42,
                    maxiter=300,
                    popsize=15
                )
                
                if result.success and result.fun < best_error:
                    # フィット品質の評価
                    predicted = self.lppls_equation(t_window, result.x, tc)
                    ss_res = np.sum((log_prices_window - predicted) ** 2)
                    ss_tot = np.sum((log_prices_window - np.mean(log_prices_window)) ** 2)
                    r_squared = 1 - (ss_res / ss_tot)
                    
                    if r_squared > best_r_squared:
                        best_error = result.fun
                        best_fit = result.x
                        best_tc = tc
                        best_r_squared = r_squared
                        
            except Exception as e:
                continue
        
        if best_fit is not None:
            # 結果の保存
            A, B, C, m, omega, phi = best_fit
            
            # 臨界時間を日付に変換
            tc_date = dates_window[0] + pd.Timedelta(days=int(best_tc - t_window[0]))
            days_to_critical = int(best_tc - t_window[-1])
            
            # 予測価格の計算
            t_extended = np.arange(t_window[0], best_tc, 1)
            predicted_log_prices = self.lppls_equation(t_extended, best_fit, best_tc)
            predicted_prices = np.exp(predicted_log_prices)
            
            # 信頼度の計算
            confidence = self._calculate_confidence(best_r_squared, m, omega, C)
            
            self.lppls_results = {
                'parameters': {
                    'A': A, 'B': B, 'C': C, 'm': m, 'omega': omega, 'phi': phi
                },
                'critical_time': best_tc,
                'critical_date': tc_date,
                'days_to_critical': days_to_critical,
                'r_squared': best_r_squared,
                'confidence': confidence,
                'window_data': {
                    't': t_window,
                    'dates': dates_window,
                    'log_prices': log_prices_window,
                    'prices': np.exp(log_prices_window)
                },
                'prediction': {
                    't': t_extended,
                    'log_prices': predicted_log_prices,
                    'prices': predicted_prices
                },
                'fit_quality': self._get_fit_quality(best_r_squared)
            }
            
            print(f"✅ LPPLS fitting completed")
            print(f"   R²: {best_r_squared:.4f}")
            print(f"   Critical date: {tc_date.date()}")
            print(f"   Days to critical: {days_to_critical}")
            print(f"   Confidence: {confidence}")
            print(f"   Fit quality: {self._get_fit_quality(best_r_squared)}")
            
            return self.lppls_results
        else:
            print("❌ LPPLS fitting failed")
            return None
    
    def _calculate_confidence(self, r_squared, m, omega, C):
        """信頼度の計算"""
        confidence_score = 0
        
        # R²による評価
        if r_squared > 0.8:
            confidence_score += 40
        elif r_squared > 0.6:
            confidence_score += 25
        elif r_squared > 0.4:
            confidence_score += 10
        
        # 臨界指数mによる評価
        if 0.1 < m < 0.9:
            confidence_score += 20
        elif 0.05 < m < 0.95:
            confidence_score += 10
        
        # 角周波数ωによる評価
        if 2 < omega < 20:
            confidence_score += 20
        elif 1 < omega < 25:
            confidence_score += 10
        
        # 振動振幅Cによる評価
        if abs(C) > 0.01:
            confidence_score += 20
        elif abs(C) > 0.005:
            confidence_score += 10
        
        return min(confidence_score, 100)
    
    def _get_fit_quality(self, r_squared):
        """フィット品質の判定"""
        if r_squared > 0.8:
            return "Excellent"
        elif r_squared > 0.6:
            return "Good"
        elif r_squared > 0.4:
            return "Fair"
        else:
            return "Poor"
    
    def plot_lppls_analysis(self):
        """LPPLS分析結果のプロット"""
        if not self.lppls_results:
            print("❌ No LPPLS results available. Please run fit_lppls_model first.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle(f'{self.symbol} - LPPLS Dragon King Analysis (10 Years)', 
                     fontsize=16, fontweight='bold')
        
        # 1. 価格チャートとLPPLS予測
        ax1 = axes[0, 0]
        
        # 全期間の価格データ
        full_dates = self.data.index
        full_prices = self.data['Close'].values
        ax1.plot(full_dates, full_prices, 'b-', linewidth=1, alpha=0.7, label='Historical Price')
        
        # フィッティングウィンドウ
        window_data = self.lppls_results['window_data']
        ax1.plot(window_data['dates'], window_data['prices'], 'b-', linewidth=2, label='Fitting Window')
        
        # LPPLS予測
        prediction = self.lppls_results['prediction']
        pred_dates = [window_data['dates'][0] + pd.Timedelta(days=int(t - prediction['t'][0])) 
                     for t in prediction['t']]
        ax1.plot(pred_dates, prediction['prices'], 'r--', linewidth=2, label='LPPLS Prediction')
        
        # 臨界点
        critical_date = self.lppls_results['critical_date']
        ax1.axvline(x=critical_date, color='red', linestyle=':', linewidth=2,
                   label=f'Critical Point ({self.lppls_results["days_to_critical"]} days)')
        
        ax1.set_title('S&P500 Price and LPPLS Prediction')
        ax1.set_ylabel('Price (USD)')
        ax1.set_yscale('log')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # X軸の日付フォーマット
        ax1.xaxis.set_major_formatter(DateFormatter('%Y\n%m'))
        ax1.xaxis.set_major_locator(mdates.YearLocator())
        ax1.xaxis.set_minor_locator(mdates.MonthLocator(interval=6))
        
        # 2. ログ価格とフィッティング
        ax2 = axes[0, 1]
        ax2.plot(window_data['dates'], window_data['log_prices'], 'b-', linewidth=2, label='Actual Log Price')
        
        # フィッティング結果
        fitted_log_prices = self.lppls_equation(
            window_data['t'], 
            list(self.lppls_results['parameters'].values()), 
            self.lppls_results['critical_time']
        )
        ax2.plot(window_data['dates'], fitted_log_prices, 'r--', linewidth=2, label='LPPLS Fit')
        
        ax2.set_title(f'Log Price Fitting (R² = {self.lppls_results["r_squared"]:.4f})')
        ax2.set_ylabel('Log Price')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # X軸の日付フォーマット
        ax2.xaxis.set_major_formatter(DateFormatter('%m/%d\n%y'))
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        
        # 3. 残差分析
        ax3 = axes[1, 0]
        residuals = window_data['log_prices'] - fitted_log_prices
        ax3.plot(window_data['dates'], residuals, 'g-', linewidth=1)
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax3.axhline(y=np.std(residuals), color='red', linestyle='--', alpha=0.7, label='+1σ')
        ax3.axhline(y=-np.std(residuals), color='red', linestyle='--', alpha=0.7, label='-1σ')
        
        ax3.set_title('Residuals Analysis')
        ax3.set_ylabel('Residuals')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # X軸の日付フォーマット
        ax3.xaxis.set_major_formatter(DateFormatter('%m/%d\n%y'))
        ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        
        # 4. パラメータ表示
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # パラメータ情報
        params = self.lppls_results['parameters']
        info_text = f"""
LPPLS Parameters:
A (Log Price Level): {params['A']:.4f}
B (Power Law Amplitude): {params['B']:.4f}
C (Oscillation Amplitude): {params['C']:.4f}
m (Critical Exponent): {params['m']:.4f}
ω (Angular Frequency): {params['omega']:.4f}
φ (Phase): {params['phi']:.4f}

Model Quality:
R²: {self.lppls_results['r_squared']:.4f}
Fit Quality: {self.lppls_results['fit_quality']}
Confidence: {self.lppls_results['confidence']}/100

Critical Point:
Date: {self.lppls_results['critical_date'].strftime('%Y-%m-%d')}
Days to Critical: {self.lppls_results['days_to_critical']}

Dragon King Risk Assessment:
{self._get_risk_assessment()}
        """
        
        ax4.text(0.05, 0.95, info_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
        
        # 結果サマリーの表示
        self._display_results_summary()
    
    def _get_risk_assessment(self):
        """リスク評価の取得"""
        confidence = self.lppls_results['confidence']
        days_to_critical = self.lppls_results['days_to_critical']
        r_squared = self.lppls_results['r_squared']
        
        if confidence > 70 and days_to_critical < 60 and r_squared > 0.7:
            return "🔴 HIGH RISK - Potential bubble detected"
        elif confidence > 50 and days_to_critical < 120 and r_squared > 0.5:
            return "🟠 MODERATE RISK - Monitor closely"
        elif confidence > 30:
            return "🟡 LOW RISK - Some bubble indicators"
        else:
            return "🟢 MINIMAL RISK - No significant bubble signals"
    
    def _display_results_summary(self):
        """結果サマリーの表示"""
        print("\n" + "="*80)
        print("🐉 LPPLS DRAGON KING ANALYSIS SUMMARY")
        print("="*80)
        print(f"Symbol: {self.symbol}")
        print(f"Analysis Period: {self.period}")
        print(f"Critical Date: {self.lppls_results['critical_date'].strftime('%Y-%m-%d')}")
        print(f"Days to Critical: {self.lppls_results['days_to_critical']}")
        print(f"Model R²: {self.lppls_results['r_squared']:.4f}")
        print(f"Confidence Score: {self.lppls_results['confidence']}/100")
        print(f"Risk Assessment: {self._get_risk_assessment()}")
        print("="*80)
    
    def run_full_analysis(self, window_size=500, tc_range_days=250):
        """完全な分析を実行"""
        print("🐉 Starting LPPLS Dragon King Analysis for S&P500")
        print("="*60)
        
        # データ取得
        if not self.fetch_data():
            return False
        
        # LPPLS モデルフィッティング
        result = self.fit_lppls_model(window_size, tc_range_days)
        
        if result:
            # 結果をプロット
            self.plot_lppls_analysis()
            return True
        else:
            print("❌ Analysis failed")
            return False

# 使用例とデモ
def demo_sp500_lppls_analysis():
    """SP500 LPPLS分析のデモ実行"""
    print("🚀 S&P500 LPPLS Dragon King Analysis Demo")
    print("="*60)
    
    # SP500の10年分データで分析
    analyzer = LPPLSAnalyzer("^GSPC", period="10y")
    
    if analyzer.run_full_analysis(window_size=500, tc_range_days=250):
        return analyzer
    else:
        print("❌ Demo analysis failed")
        return None

if __name__ == "__main__":
    # 必要なライブラリのインストール（Colab用）
    if IN_COLAB:
        print("📦 Installing required packages...")
        import subprocess
        import sys
        
        packages = ["yfinance", "scipy"]
        for package in packages:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    # デモ実行
    demo_result = demo_sp500_lppls_analysis()