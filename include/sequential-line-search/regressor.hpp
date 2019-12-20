#ifndef SEQUENTIAL_LINE_SEARCH_REGRESSOR_HPP
#define SEQUENTIAL_LINE_SEARCH_REGRESSOR_HPP

#include <Eigen/Core>
#include <vector>

namespace sequential_line_search
{
    enum class KernelType
    {
        ArdSquaredExponentialKernel,
        ArdMatern52Kernel,
    };

    class Regressor
    {
    public:
        Regressor(const KernelType kernel_type);
        virtual ~Regressor() {}

        unsigned GetNumDims() const { return GetLargeX().rows(); }

        virtual double PredictMu(const Eigen::VectorXd& x) const    = 0;
        virtual double PredictSigma(const Eigen::VectorXd& x) const = 0;

        virtual Eigen::VectorXd PredictMuDerivative(const Eigen::VectorXd& x) const    = 0;
        virtual Eigen::VectorXd PredictSigmaDerivative(const Eigen::VectorXd& x) const = 0;

        virtual const Eigen::VectorXd& GetKernelHyperparams() const = 0;
        virtual double                 GetNoiseHyperparam() const   = 0;

        virtual const Eigen::MatrixXd& GetLargeX() const = 0;
        virtual const Eigen::VectorXd& GetSmallY() const = 0;

        Eigen::VectorXd PredictMaximumPointFromData() const;

    private:
        double (*m_kernel)(const Eigen::VectorXd&, const Eigen::VectorXd&, const Eigen::VectorXd&);
        Eigen::VectorXd (*m_kernel_theta_derivative)(const Eigen::VectorXd&,
                                                     const Eigen::VectorXd&,
                                                     const Eigen::VectorXd&);
        Eigen::VectorXd (*m_kernel_first_arg_derivative)(const Eigen::VectorXd&,
                                                         const Eigen::VectorXd&,
                                                         const Eigen::VectorXd&);
    };

    // k
    Eigen::VectorXd
    CalcSmallK(const Eigen::VectorXd& x, const Eigen::MatrixXd& X, const Eigen::VectorXd& kernel_hyperparameters);

    // K_y = K_f + sigma^{2} I
    Eigen::MatrixXd
    CalcLargeKY(const Eigen::MatrixXd& X, const Eigen::VectorXd& kernel_hyperparameters, const double noise_level);

    // K_f
    Eigen::MatrixXd CalcLargeKF(const Eigen::MatrixXd& X, const Eigen::VectorXd& kernel_hyperparameters);

    // partial k / partial x
    Eigen::MatrixXd CalcSmallKSmallXDerivative(const Eigen::VectorXd& x,
                                               const Eigen::MatrixXd& X,
                                               const Eigen::VectorXd& kernel_hyperparameters);

    // partial K_y / partial theta
    std::vector<Eigen::MatrixXd> CalcLargeKYThetaDerivative(const Eigen::MatrixXd& X,
                                                            const Eigen::VectorXd& kernel_hyperparameters);

    // partial K_y / partial sigma^{2}
    Eigen::MatrixXd CalcLargeKYNoiseLevelDerivative(const Eigen::MatrixXd& X,
                                                    const Eigen::VectorXd& kernel_hyperparameters,
                                                    const double           noise_level);

} // namespace sequential_line_search

#endif // SEQUENTIAL_LINE_SEARCH_REGRESSOR_HPP
