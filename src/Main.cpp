#include "itkImageFileReader.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkBinaryThresholdImageFilter.h"
#include "itkImageFileWriter.h"

#include "itkEllipseSpatialObject.h" 
#include "itkSpatialObjectToImageFilter.h"
#include "itkImageToSpatialObjectRegistrationMethod.h"
#include "itkOnePlusOneEvolutionaryOptimizer.h"
#include "itkNormalVariateGenerator.h"
#include "itkSpatialObjectWriter.h"

#include "itkImageRegistrationMethodv4.h"
#include "itkMeanSquaresImageToImageMetricv4.h"
#include "itkRegularStepGradientDescentOptimizerv4.h"
#include "itkCenteredTransformInitializer.h"
#include "itkCenteredSimilarity2DTransform.h"
#include "itkResampleImageFilter.h"
#include "itkCastImageFilter.h"
#include "itkSubtractImageFilter.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkIdentityTransform.h"

#include "itkMattesMutualInformationImageToImageMetric.h"
#include "itkCorrelationImageToImageMetricv4.h"
#include <itkJointHistogramMutualInformationImageToImageMetricv4.h>
#include "itkOnePlusOneEvolutionaryOptimizerv4.h"

#include "itkBinaryImageToLabelMapFilter.h"
#include "itkLabelMapOverlayImageFilter.h"
#include "itkBilateralImageFilter.h"
#include "itkAffineTransform.h"

#include "QuickView.h"

#include <math.h>
#include <iostream>

typedef unsigned char                                                   UCPixelType;
typedef itk::RGBPixel< UCPixelType >                                    RGBPixelType;
typedef itk::Image< UCPixelType, 2>                                     UCImageType;
typedef itk::Image< double, 2 >                                         DoubleImageType;
typedef itk::Image< float, 2 >                                          FloatImageType;
typedef itk::Image< RGBPixelType, 2 >                                   RGBImageType;
typedef itk::ImageFileReader<UCImageType>                               ImageReaderType;
typedef itk::ImageFileReader<FloatImageType>                            FloatImageReaderType;
typedef itk::CastImageFilter< UCImageType, RGBImageType>                CastToRGBFilterType;

#include "itkCommand.h"

namespace itk
{
  //  The following section of code implements a Command observer
  //  that will monitor the evolution of the registration process.
  //
class CommandIterationUpdate : public itk::Command
  {
  public:
    typedef  CommandIterationUpdate   Self;
    typedef  itk::Command             Superclass;
    typedef itk::SmartPointer<Self>   Pointer;
    itkNewMacro( Self );
  protected:
    CommandIterationUpdate() { m_LastMetricValue = 0.0; };
  public:
    //typedef itk::RegularStepGradientDescentOptimizerv4<double> OptimizerType;
    typedef itk::OnePlusOneEvolutionaryOptimizerv4<double>     OptimizerType;
    typedef   const OptimizerType *                            OptimizerPointer;
    void Execute( itk::Object *caller, const itk::EventObject & event ) ITK_OVERRIDE
      {
      Execute( ( const itk::Object * )caller, event );
      }
    void Execute( const itk::Object * object, const itk::EventObject & event ) ITK_OVERRIDE
      {
      OptimizerPointer optimizer = static_cast< OptimizerPointer >( object );
      if( !itk::IterationEvent().CheckEvent( &event ) )
        {
        return;
        }
      double currentValue = optimizer->GetValue();
      // Only print out when the Metric value changes
      if( std::fabs( m_LastMetricValue - currentValue ) > 1e-7 )
        {
        std::cout << optimizer->GetCurrentIteration() << "   ";
        std::cout << optimizer->GetValue() << "   ";
        std::cout << optimizer->GetCurrentPosition() << std::endl;
        m_LastMetricValue = currentValue;
        }
      }
  private:
    double m_LastMetricValue;
  };
}//end of itk namespace

int ImageRegistrationMethod( UCImageType::Pointer inputImage, unsigned int number, int threshold )
{
  QuickView viewer;
  viewer.AddImage<UCImageType>( inputImage );

  const    unsigned int    Dimension = 2;
  typedef  float           PixelType;
  typedef itk::Image< PixelType, Dimension >  FixedImageType;
  typedef itk::Image< PixelType, Dimension >  MovingImageType;

  // Define Fixed Image
  typedef itk::CastImageFilter< UCImageType, FixedImageType >      CastToFloatFilterType;
  CastToFloatFilterType::Pointer castFilter = CastToFloatFilterType::New();
  castFilter->SetInput( inputImage );
  castFilter->Update();

  /*typedef itk::RescaleIntensityImageFilter<
    FixedImageType,
    FixedImageType >   RescalerFloatType;
  RescalerFloatType::Pointer intensityRescaler = RescalerFloatType::New();
  intensityRescaler->SetInput( castFilter->GetOutput() );
  intensityRescaler->SetOutputMinimum( 0 );
  intensityRescaler->SetOutputMaximum( 255 );*/

  typedef itk::DiscreteGaussianImageFilter<FixedImageType, FixedImageType >    DiscretGaussianFilterType;
  DiscretGaussianFilterType::Pointer gaussianFilter = DiscretGaussianFilterType::New();
  float variance = 20.0;
  gaussianFilter->SetInput( castFilter->GetOutput() );
  gaussianFilter->SetVariance( variance );
  gaussianFilter->Update();

  viewer.AddImage<FixedImageType>( gaussianFilter->GetOutput(), true, "inputImage - gaussian filter" );

  // Smooth while preserving the edges -- not used
  typedef itk::BilateralImageFilter<FixedImageType, FixedImageType >    BilateralFilterType;
  BilateralFilterType::Pointer bilateralFilter = BilateralFilterType::New();
  bilateralFilter->SetInput( castFilter->GetOutput() );
  bilateralFilter->SetDomainSigma( 4 );
  bilateralFilter->SetRangeSigma( 20 );

  //viewer.AddImage<FixedImageType>( bilateralFilter->GetOutput(), true, "inputImage - bilateral filter" );

  typedef itk::BinaryThresholdImageFilter<
    FixedImageType, FixedImageType >  ThresholdFilterType;
  ThresholdFilterType::Pointer thresholdFilter = ThresholdFilterType::New();
  thresholdFilter->SetInput( gaussianFilter->GetOutput() );
  thresholdFilter->SetOutsideValue( 255 );
  thresholdFilter->SetInsideValue( 0 );
  thresholdFilter->SetLowerThreshold( threshold );
  thresholdFilter->SetUpperThreshold( 255 );

  FixedImageType::Pointer fixedImage = thresholdFilter->GetOutput();
  viewer.AddImage<FixedImageType>( fixedImage, true, "fixedImage" );

  UCImageType::SizeType input_size = inputImage->GetLargestPossibleRegion().GetSize();

  // Define Moving Image = ellipse
  typedef itk::EllipseSpatialObject< 2 >   EllipseType;
  EllipseType::Pointer ellipse = EllipseType::New();
  
  EllipseType::ArrayType radius;
  radius[ 0 ] = 100;
  radius[ 1 ] = 85;
  ellipse->SetRadius( radius );

  EllipseType::TransformType::OffsetType offset;
  offset[ 0 ] = input_size[ 0 ] / 2;
  offset[ 1 ] = input_size[ 1 ] / 2;
  ellipse->GetObjectToParentTransform()->SetOffset( offset );
  ellipse->ComputeObjectToWorldTransform();

  typedef itk::SpatialObjectToImageFilter< EllipseType, FloatImageType >
    SpatialObjectToImageFilterType;
  SpatialObjectToImageFilterType::Pointer spatialObjectToImageFilter =
    SpatialObjectToImageFilterType::New();
  spatialObjectToImageFilter->SetInput( ellipse );
  spatialObjectToImageFilter->SetSize( fixedImage->GetLargestPossibleRegion().GetSize() );
  spatialObjectToImageFilter->Update();
  ellipse->SetDefaultInsideValue( 255 );
  ellipse->SetDefaultOutsideValue( 0 );
  spatialObjectToImageFilter->SetUseObjectValue( true );
  spatialObjectToImageFilter->SetOutsideValue( 0 );

  MovingImageType::Pointer movingImage = spatialObjectToImageFilter->GetOutput();
  viewer.AddImage<MovingImageType>( movingImage, true, "Original movingImage" );
  //viewer.Visualize();

  typedef itk::CenteredSimilarity2DTransform< double >              TransformType;
  //typedef itk::AffineTransform< double, 2 >                         TransformType;
  typedef itk::OnePlusOneEvolutionaryOptimizerv4< double >          OptimizerType;
  typedef itk::MattesMutualInformationImageToImageMetricv4<
    FixedImageType, MovingImageType >                               MetricType;
  typedef itk::ImageRegistrationMethodv4< FixedImageType,
    MovingImageType,
    TransformType >                                                 RegistrationType;

  TransformType::Pointer      transform = TransformType::New();  
  OptimizerType::Pointer      optimizer = OptimizerType::New();
  MetricType::Pointer         metric = MetricType::New();
  RegistrationType::Pointer   registration = RegistrationType::New();
  
  registration->SetMetric( metric );
  unsigned int numberOfBins = 50;
  metric->SetNumberOfHistogramBins( numberOfBins );
  
  registration->SetOptimizer( optimizer );

  registration->SetFixedImage( fixedImage );
  registration->SetMovingImage( movingImage );

  typedef itk::CenteredTransformInitializer<
    TransformType,
    FixedImageType,
    MovingImageType > TransformInitializerType;
  TransformInitializerType::Pointer initializer = TransformInitializerType::New();
  initializer->SetTransform( transform );
  initializer->SetFixedImage( fixedImage );
  initializer->SetMovingImage( movingImage );
  initializer->MomentsOn();
  initializer->InitializeTransform();
  std::cout << "Initial Parameters  : " << transform->GetParameters() << std::endl;

  typedef itk::ResampleImageFilter< MovingImageType,
    FixedImageType > ResampleFilterType2;
  ResampleFilterType2::Pointer resampler2 = ResampleFilterType2::New();
  resampler2->SetTransform( transform );
  resampler2->SetInput( movingImage );
  resampler2->SetSize( fixedImage->GetLargestPossibleRegion().GetSize() );
  resampler2->SetOutputOrigin( fixedImage->GetOrigin() );
  resampler2->SetOutputSpacing( fixedImage->GetSpacing() );
  resampler2->SetOutputDirection( fixedImage->GetDirection() );
  resampler2->SetDefaultPixelValue( 100 );
  viewer.AddImage<MovingImageType>( resampler2->GetOutput(), true, "movingImage - initial tranform" );

  registration->SetInitialTransform( transform );
  registration->InPlaceOn();

  typedef OptimizerType::ScalesType       OptimizerScalesType;
  OptimizerScalesType optimizerScales( transform->GetNumberOfParameters() );
  optimizerScales[ 0 ] = 1000.0;
  optimizerScales[ 1 ] = 10000.0;
  optimizerScales[ 2 ] = 10.0;
  optimizerScales[ 3 ] = 10.0;
  optimizerScales[ 4 ] = 10.0;
  optimizerScales[ 5 ] = 10.0;

  optimizer->SetScales( optimizerScales );

  /*typedef itk::RegistrationParameterScalesFromPhysicalShift<MetricType> ScalesEstimatorType;
  ScalesEstimatorType::Pointer scalesEstimator = ScalesEstimatorType::New();
  scalesEstimator->SetMetric( metric );
  scalesEstimator->SetTransformForward( true );
  optimizer->SetScalesEstimator( scalesEstimator );*/

  typedef itk::Statistics::NormalVariateGenerator  GeneratorType;
  GeneratorType::Pointer generator = GeneratorType::New();
  optimizer->SetNormalVariateGenerator( generator );
  optimizer->Initialize( 100 );
  optimizer->SetEpsilon( 1.0 );
  optimizer->SetMaximumIteration( 4000 );

  // Create the Command observer and register it with the optimizer.
  itk::CommandIterationUpdate::Pointer observer = itk::CommandIterationUpdate::New();
  optimizer->AddObserver( itk::IterationEvent(), observer );

  // One level registration process without shrinking and smoothing.
  const unsigned int numberOfLevels = 1;
  RegistrationType::ShrinkFactorsArrayType shrinkFactorsPerLevel;
  shrinkFactorsPerLevel.SetSize( 1 );
  shrinkFactorsPerLevel[ 0 ] = 1;
  RegistrationType::SmoothingSigmasArrayType smoothingSigmasPerLevel;
  smoothingSigmasPerLevel.SetSize( 1 );
  smoothingSigmasPerLevel[ 0 ] = 0;
  registration->SetMetricSamplingPercentage( 1.0 );
  registration->SetNumberOfLevels( numberOfLevels );
  registration->SetSmoothingSigmasPerLevel( smoothingSigmasPerLevel );
  registration->SetShrinkFactorsPerLevel( shrinkFactorsPerLevel );

  try
    {
    registration->Update();
    std::cout<< "Optimizer scale = " << optimizer->GetScales() << std::endl;
    std::cout << "Optimizer stop condition: "
      << registration->GetOptimizer()->GetStopConditionDescription()
      << std::endl;
    }
  catch( itk::ExceptionObject & err )
    {
    std::cerr << "ExceptionObject caught !" << std::endl;
    std::cerr << err << std::endl;
    return EXIT_FAILURE;
    }
  // Get the final parameters - Similarity transform
  TransformType::ParametersType finalParameters = transform->GetParameters();
  const double finalScale = finalParameters[ 0 ];
  const double finalAngle = finalParameters[ 1 ];
  const double finalRotationCenterX = finalParameters[ 2 ];
  const double finalRotationCenterY = finalParameters[ 3 ];
  const double finalTranslationX = finalParameters[ 4 ];
  const double finalTranslationY = finalParameters[ 5 ];
  const unsigned int numberOfIterations = optimizer->GetCurrentIteration();
  const double bestValue = optimizer->GetValue();
  
  // Print out results
  const double finalAngleInDegrees = finalAngle * 180.0 / itk::Math::pi;
  std::cout << std::endl;
  std::cout << "Result = " << std::endl;
  std::cout << " Scale         = " << finalScale << std::endl;
  std::cout << " Angle (radians) " << finalAngle << std::endl;
  std::cout << " Angle (degrees) " << finalAngleInDegrees << std::endl;
  std::cout << " Center X      = " << finalRotationCenterX << std::endl;
  std::cout << " Center Y      = " << finalRotationCenterY << std::endl;
  std::cout << " Translation X = " << finalTranslationX << std::endl;
  std::cout << " Translation Y = " << finalTranslationY << std::endl;
  std::cout << " Iterations    = " << numberOfIterations << std::endl;
  std::cout << " Metric value  = " << bestValue << std::endl;

  // Get the final parameters - Affine transform
  /*const TransformType::ParametersType finalParameters =
    registration->GetOutput()->Get()->GetParameters();
  const double finalRotationCenterX = transform->GetCenter()[ 0 ];
  const double finalRotationCenterY = transform->GetCenter()[ 1 ];
  const double finalTranslationX = finalParameters[ 4 ];
  const double finalTranslationY = finalParameters[ 5 ];
  const unsigned int numberOfIterations = optimizer->GetCurrentIteration();
  const double bestValue = optimizer->GetValue();
  
  // Print out results
  std::cout << "Result = " << std::endl;
  std::cout << " Center X      = " << finalRotationCenterX << std::endl;
  std::cout << " Center Y      = " << finalRotationCenterY << std::endl;
  std::cout << " Translation X = " << finalTranslationX << std::endl;
  std::cout << " Translation Y = " << finalTranslationY << std::endl;
  std::cout << " Iterations    = " << numberOfIterations << std::endl;
  std::cout << " Metric value  = " << bestValue << std::endl;

  //Compute the rotation angle and scaling from SVD of the matrix
  // VNL returns the eigenvalues ordered from largest to smallest.
  vnl_matrix<double> p( 2, 2 );
  p[ 0 ][ 0 ] = ( double )finalParameters[ 0 ];
  p[ 0 ][ 1 ] = ( double )finalParameters[ 1 ];
  p[ 1 ][ 0 ] = ( double )finalParameters[ 2 ];
  p[ 1 ][ 1 ] = ( double )finalParameters[ 3 ];
  vnl_svd<double> svd( p );
  vnl_matrix<double> r( 2, 2 );
  r = svd.U() * vnl_transpose( svd.V() );
  double angle = std::asin( r[ 1 ][ 0 ] );
  const double angleInDegrees = angle * 180.0 / itk::Math::pi;
  std::cout << " Scale 1         = " << svd.W( 0 ) << std::endl;
  std::cout << " Scale 2         = " << svd.W( 1 ) << std::endl;
  std::cout << " Angle (degrees) = " << angleInDegrees << std::endl;*/

  // Apply the final transform to the moving image
  typedef itk::ResampleImageFilter< MovingImageType,
    FixedImageType > ResampleFilterType;
  ResampleFilterType::Pointer resampler = ResampleFilterType::New();
  resampler->SetTransform( transform );
  resampler->SetInput( movingImage );
  resampler->SetSize( fixedImage->GetLargestPossibleRegion().GetSize() );
  resampler->SetOutputOrigin( fixedImage->GetOrigin() );
  resampler->SetOutputSpacing( fixedImage->GetSpacing() );
  resampler->SetOutputDirection( fixedImage->GetDirection() );
  resampler->SetDefaultPixelValue( 100 );

  typedef  unsigned char                            OutputPixelType;
  typedef itk::Image< OutputPixelType, Dimension >  OutputImageType;
  typedef itk::CastImageFilter< FixedImageType, OutputImageType >
    CastFilterType;
  CastFilterType::Pointer  caster = CastFilterType::New();
  caster->SetInput( resampler->GetOutput() );
  caster->Update();
  viewer.AddImage<OutputImageType>( caster->GetOutput(), true, "Resampled movingImage" );

  // Compute the difference image between the
  // fixed and moving image before registration.
  typedef itk::SubtractImageFilter<
    FixedImageType,
    FixedImageType,
    FixedImageType >    DifferenceFilterType;
  DifferenceFilterType::Pointer difference_before = DifferenceFilterType::New();
  difference_before->SetInput1( fixedImage );
  difference_before->SetInput2( movingImage );

  typedef itk::RescaleIntensityImageFilter<
    FixedImageType,
    OutputImageType >   RescalerType;
  RescalerType::Pointer intensityRescaler_b = RescalerType::New();
  intensityRescaler_b->SetInput( difference_before->GetOutput() );
  intensityRescaler_b->SetOutputMinimum( 0 );
  intensityRescaler_b->SetOutputMaximum( 255 );

  viewer.AddImage<OutputImageType>( intensityRescaler_b->GetOutput(), true, "Difference between the fixed and moving images before registration" );

  // Compute the difference image between the
  // binary fixed image and the resampled moving image.
  DifferenceFilterType::Pointer difference_after = DifferenceFilterType::New();
  difference_after->SetInput1( fixedImage );
  difference_after->SetInput2( resampler->GetOutput() );

  RescalerType::Pointer intensityRescaler_a = RescalerType::New();
  intensityRescaler_a->SetInput( difference_after->GetOutput() );
  intensityRescaler_a->SetOutputMinimum( 0 );
  intensityRescaler_a->SetOutputMaximum( 255 );

  viewer.AddImage<OutputImageType>( intensityRescaler_a->GetOutput(), true, "Difference between the fixed and resampled moving image"  );

  // Overlay the ellipse found to the input image
  typedef itk::BinaryImageToLabelMapFilter<OutputImageType> BinaryImageToLabelMapFilterType;
  BinaryImageToLabelMapFilterType::Pointer binaryImageToLabelMapFilter = BinaryImageToLabelMapFilterType::New();
  binaryImageToLabelMapFilter->SetInput( caster->GetOutput() );
  binaryImageToLabelMapFilter->Update();

  typedef itk::LabelMapOverlayImageFilter< BinaryImageToLabelMapFilterType::OutputImageType, OutputImageType > LabelMapOverlayFilterType;
  LabelMapOverlayFilterType::Pointer labelMapOverlayFilter = LabelMapOverlayFilterType::New();
  labelMapOverlayFilter->SetInput( binaryImageToLabelMapFilter->GetOutput() );
  labelMapOverlayFilter->SetFeatureImage( inputImage );
  labelMapOverlayFilter->SetOpacity( 0.5 );

  viewer.AddImage<LabelMapOverlayFilterType::OutputImageType>( labelMapOverlayFilter->GetOutput(), true, "label image" );

  typedef itk::ImageFileWriter< LabelMapOverlayFilterType::OutputImageType >    WriterType;
  WriterType::Pointer imageWriter = WriterType::New();
  imageWriter->SetInput( labelMapOverlayFilter->GetOutput() );
  
  std::string filename = "C:\\MyProjects\\Results_IntracranialPressure\\Results_6\\" + std::to_string( number ) + "_output.png";
  imageWriter->SetFileName( filename.c_str() );
  imageWriter->Update();

  viewer.SetNumberOfColumns( 3 );
  viewer.Visualize();
  return EXIT_SUCCESS;
}


int main()
  {
  unsigned int number = 0;
  for( number = 18; number < 19; number++ )
    {
    ImageReaderType::Pointer reader = ImageReaderType::New();
    std::cout << std::endl;
    std::cout << "Image " << number << std::endl;
    std::string filename = "C:\\MyProjects\\IntracranialPressure\\ONSD-Data\\0" + std::to_string( number ) + ".png";
    reader->SetFileName( filename.c_str() );
    try
      {
      reader->Update();
      }
    catch( itk::ExceptionObject & excep )
      {
      std::cerr << "Exception caught !" << std::endl;
      std::cerr << excep << std::endl;
      return EXIT_FAILURE;
      }

    int threshold = 15;
    if( number == 11 || number == 14 || number == 15 || number == 16 )
      {
      threshold = 50;
      }
    if( number == 17 )
      {
      threshold = 90;
      }

    int success = ImageRegistrationMethod( reader->GetOutput(), number, threshold );
    if( success != EXIT_SUCCESS )
      {
      std::cerr << "Error in the selected method" << std::endl;
      }
    }

  return 0;
  }

