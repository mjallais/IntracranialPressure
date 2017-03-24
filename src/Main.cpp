#define _USE_MATH_DEFINES

// Hough Method
#include "itkCannyEdgeDetectionImageFilter.h"
#include "itkCastImageFilter.h"
#include "itkDiscreteGaussianImageFilter.h"
#include "itkHoughTransform2DCirclesImageFilter.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkStatisticsImageFilter.h"

// Parametric Snake Method
#include "itkGeodesicActiveContourLevelSetImageFilter.h"
#include "itkCurvatureAnisotropicDiffusionImageFilter.h"
#include "itkGradientMagnitudeRecursiveGaussianImageFilter.h"
#include "itkSigmoidImageFilter.h"
#include "itkFastMarchingImageFilter.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkBinaryThresholdImageFilter.h"
#include "itkImageFileWriter.h"


// SpatialObjectImageRegistration Method
#include "itkEllipseSpatialObject.h" 
#include "itkGroupSpatialObject.h"
#include "itkSpatialObjectToImageFilter.h"
#include "itkImageToSpatialObjectRegistrationMethod.h"
#include "itkLinearInterpolateImageFunction.h"
#include "itkEuler2DTransform.h"
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
template < class TOptimizer >
class IterationCallback : public Command
  {
  public:
    typedef IterationCallback             Self;
    typedef Command                       Superclass;
    typedef SmartPointer<Self>            Pointer;
    typedef SmartPointer<const Self>      ConstPointer;
    itkTypeMacro( IterationCallback, Superclass );
    itkNewMacro( Self );
    typedef    TOptimizer     OptimizerType;
    void SetOptimizer( OptimizerType * optimizer )
      {
      m_Optimizer = optimizer;
      m_Optimizer->AddObserver( IterationEvent(), this );
      }
    void Execute( Object *caller, const EventObject & event )
      {
      Execute( ( const Object * )caller, event );
      }
    void Execute( const Object *, const EventObject & event )
      {
      if( typeid( event ) == typeid( StartEvent ) )
        {
        std::cout << std::endl << "Position              Value";
        std::cout << std::endl << std::endl;
        }
      else if( typeid( event ) == typeid( IterationEvent ) )
        {
        std::cout << m_Optimizer->GetCurrentIteration() << "   ";
        std::cout << m_Optimizer->GetValue() << "   ";
        std::cout << m_Optimizer->GetCurrentPosition() << std::endl;
        }
      else if( typeid( event ) == typeid( EndEvent ) )
        {
        std::cout << std::endl << std::endl;
        std::cout << "After " << m_Optimizer->GetCurrentIteration();
        std::cout << "  iterations " << std::endl;
        std::cout << "Solution is    = " << m_Optimizer->GetCurrentPosition();
        std::cout << std::endl;
        }
      }
  protected:
    IterationCallback() {};
    WeakPointer<OptimizerType>   m_Optimizer;
  };

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

template <typename TFixedImage, typename TMovingSpatialObject>
class SimpleImageToSpatialObjectMetric :
  public ImageToSpatialObjectMetric<TFixedImage, TMovingSpatialObject>
  {
  public:
    typedef SimpleImageToSpatialObjectMetric    Self;
    typedef ImageToSpatialObjectMetric<TFixedImage, TMovingSpatialObject>
      Superclass;
    typedef SmartPointer<Self>             Pointer;
    typedef SmartPointer<const Self>       ConstPointer;
    typedef Point<double, 2>               PointType;
    typedef std::list<PointType>                PointListType;
    typedef TMovingSpatialObject                MovingSpatialObjectType;
    typedef typename Superclass::ParametersType ParametersType;
    typedef typename Superclass::DerivativeType DerivativeType;
    typedef typename Superclass::MeasureType    MeasureType;
    itkNewMacro( Self );
    itkTypeMacro( SimpleImageToSpatialObjectMetric, ImageToSpatialObjectMetric );
    itkStaticConstMacro( ParametricSpaceDimension, unsigned int, 6 );
    void SetMovingSpatialObject( const MovingSpatialObjectType * object )
      {
      if( !this->m_FixedImage )
        {
        std::cout << "Please set the image before the moving spatial object" << std::endl;
        return;
        }
      this->m_MovingSpatialObject = object;
      m_PointList.clear();
      typedef ImageRegionConstIteratorWithIndex<TFixedImage> myIteratorType;
      myIteratorType it( this->m_FixedImage, this->m_FixedImage->GetBufferedRegion() );
      Point<double, 2> point;
      while( !it.IsAtEnd() )
        {
        this->m_FixedImage->TransformIndexToPhysicalPoint( it.GetIndex(), point );
        if( this->m_MovingSpatialObject->IsInside( point ) )
          {
          m_PointList.push_back( point );
          }
        ++it;
        }
      std::cout << "Number of points in the metric = " << static_cast< unsigned long >( m_PointList.size() ) << std::endl;
      }
    unsigned int GetNumberOfParameters( void ) const { return ParametricSpaceDimension; }
    void GetDerivative( const ParametersType &, DerivativeType & ) const
      {
      return;
      }
    MeasureType    GetValue( const ParametersType & parameters ) const
      {
      double value;
      this->m_Transform->SetParameters( parameters );
      value = 0;
      for( PointListType::const_iterator it = m_PointList.begin();
        it != m_PointList.end(); ++it )
        {
        PointType transformedPoint = this->m_Transform->TransformPoint( *it );
        if( this->m_Interpolator->IsInsideBuffer( transformedPoint ) )
          {
          value += this->m_Interpolator->Evaluate( transformedPoint );
          }
        }
      return value;
      }
    void GetValueAndDerivative( const ParametersType & parameters,
      MeasureType & Value, DerivativeType  & Derivative ) const
      {
      Value = this->GetValue( parameters );
      this->GetDerivative( parameters, Derivative );
      }
  private:
    PointListType m_PointList;
  };
}//end of itk namespace

void HoughMethod( UCImageType::Pointer inputImage )
  {
  typedef float                                                               AccumulatorPixelType;
  typedef itk::DiscreteGaussianImageFilter<UCImageType, UCImageType >         DiscretGaussianFilterType;
  typedef itk::CastImageFilter< UCImageType, DoubleImageType >                CastToDoubleFilterType;
  typedef itk::CannyEdgeDetectionImageFilter< DoubleImageType, DoubleImageType >
    CannyFilterType;
  typedef itk::RescaleIntensityImageFilter< DoubleImageType, UCImageType >    RescaleFilterType;
  typedef itk::HoughTransform2DCirclesImageFilter<UCPixelType, AccumulatorPixelType>
    HoughTransformFilterType;
  typedef itk::Image< AccumulatorPixelType, 2 >                               AccumulatorImageType;


  QuickView viewer;
  UCImageType::Pointer previous = inputImage;

  UCImageType::RegionType max_region = inputImage->GetLargestPossibleRegion();
  UCImageType::SizeType size = max_region.GetSize();
  std::cout << "size = " << size << std::endl;

  CastToRGBFilterType::Pointer toRGBFilter = CastToRGBFilterType::New();
  toRGBFilter->SetInput( inputImage );
  toRGBFilter->Update();

  RGBImageType::Pointer image = toRGBFilter->GetOutput();

  // 2 : Smoothing
  /*std::cout << "Smoothing" << std::endl;
  float variance = 20.0;

  DiscretGaussianFilterType::Pointer gaussianFilter = DiscretGaussianFilterType::New();
  gaussianFilter->SetInput( previous );
  gaussianFilter->SetVariance( variance );
  previous = gaussianFilter->GetOutput();
  viewer.AddImage<ImageType>( previous );*/

  // 3 : Edge Detection
  std::cout << "Edge detection" << std::endl;
  CastToDoubleFilterType::Pointer toDoubeFilter = CastToDoubleFilterType::New();
  toDoubeFilter->SetInput( previous );

  CannyFilterType::Pointer cannyFilter = CannyFilterType::New();
  cannyFilter->SetVariance( 45.0 );
  cannyFilter->SetLowerThreshold( 2.0 );
  cannyFilter->SetUpperThreshold( 5.0 );

  cannyFilter->SetInput( toDoubeFilter->GetOutput() );
  std::cout << "upperthreshold = " << cannyFilter->GetUpperThreshold() << std::endl;
  std::cout << "lowerthreshold = " << cannyFilter->GetLowerThreshold() << std::endl;
  std::cout << "variance = " << cannyFilter->GetVariance() << std::endl;

  RescaleFilterType::Pointer rescaleFilter = RescaleFilterType::New();
  rescaleFilter->SetInput( cannyFilter->GetOutput() );
  rescaleFilter->SetOutputMinimum( 0 );
  rescaleFilter->SetOutputMaximum( 255 );

  previous = rescaleFilter->GetOutput();
  //viewer.AddImage<ImageType>( previous );

  // Smooth the edges
  float variance = 5.0;
  DiscretGaussianFilterType::Pointer gaussianFilter = DiscretGaussianFilterType::New();
  gaussianFilter->SetInput( previous );
  gaussianFilter->SetVariance( variance );
  previous = gaussianFilter->GetOutput();
  viewer.AddImage<UCImageType>( previous );

  // 4 : Hough Transform
  std::cout << "Hough transform" << std::endl;
  HoughTransformFilterType::Pointer houghFilter = HoughTransformFilterType::New();
  houghFilter->SetInput( previous );

  int nb_circles = 10;
  houghFilter->SetNumberOfCircles( nb_circles );
  houghFilter->SetMinimumRadius( size[ 0 ] / 3 );
  houghFilter->SetMaximumRadius( size[ 0 ] / 2 );
  houghFilter->SetVariance( 5 );
  houghFilter->SetSweepAngle( 1 );
  houghFilter->SetThreshold( 0 );
  //houghFilter->SetSigmaGradient(2);
  houghFilter->Update();

  AccumulatorImageType::Pointer localAccumulator = houghFilter->GetOutput();
  HoughTransformFilterType::CirclesListType circles;
  circles = houghFilter->GetCircles( nb_circles );

  RGBImageType::PixelType pixel;
  pixel.SetRed( 255 );
  pixel.SetGreen( 0 );
  pixel.SetBlue( 0 );

  RGBImageType::IndexType localIndex;
  typedef HoughTransformFilterType::CirclesListType CirclesListType;
  CirclesListType::const_iterator itCircles = circles.begin();
  while( itCircles != circles.end() )
    {
    for( double angle = 0; angle <= 2 * vnl_math::pi; angle += vnl_math::pi / 90.0 )
      {
      for( int i = 0; i < 5; i++ )
        {
        localIndex[ 0 ] =
          ( long int )( ( *itCircles )->GetObjectToParentTransform()->GetOffset()[ 0 ]
            + ( ( *itCircles )->GetRadius()[ 0 ] + i ) * vcl_cos( angle ) );
        localIndex[ 1 ] =
          ( long int )( ( *itCircles )->GetObjectToParentTransform()->GetOffset()[ 1 ]
            + ( ( *itCircles )->GetRadius()[ 0 ] + i ) * vcl_sin( angle ) );
        RGBImageType::RegionType outputRegion = image->GetLargestPossibleRegion();
        if( outputRegion.IsInside( localIndex ) )
          {
          previous->SetPixel( localIndex, 255 );
          image->SetPixel( localIndex, pixel );
          }
        }
      }
    itCircles++;
    }

  viewer.AddImage<UCImageType>( previous );
  viewer.AddImage<RGBImageType>( image );

  viewer.Visualize();

  return;
  }

int ParametricSnakeMethod( UCImageType::Pointer inputImage )
  {
  typedef itk::ImageFileWriter< UCImageType >                         WriterType;
  typedef itk::ImageFileWriter< FloatImageType >                      InternalWriterType;

  QuickView viewer;

  UCImageType::RegionType max_region = inputImage->GetLargestPossibleRegion();
  UCImageType::SizeType size = max_region.GetSize();
  std::cout << "inputImage size = " << size << std::endl;

  viewer.AddImage( inputImage.GetPointer() );

  const int seedPosX = size[ 0 ] / 2;
  const int seedPosY = size[ 1 ] / 2;

  const double initialDistance = 5.0;
  const double sigma = 1.0;
  const double alpha = -0.5;
  const double beta = 3.0;
  const double propagationScaling = 2.0;
  const double numberOfIterations = 3000.0;
  const double seedValue = -initialDistance;

  const unsigned int Dimension = 2;

  typedef  itk::CurvatureAnisotropicDiffusionImageFilter< UCImageType, FloatImageType > SmoothingFilterType;
  SmoothingFilterType::Pointer smoothing = SmoothingFilterType::New();
  smoothing->SetTimeStep( 0.125 );
  smoothing->SetNumberOfIterations( 5 );
  smoothing->SetConductanceParameter( 9.0 );
  smoothing->SetInput( inputImage );

  typedef  itk::GradientMagnitudeRecursiveGaussianImageFilter< FloatImageType, FloatImageType > GradientFilterType;
  GradientFilterType::Pointer  gradientMagnitude = GradientFilterType::New();
  gradientMagnitude->SetSigma( sigma );
  gradientMagnitude->SetInput( smoothing->GetOutput() );

  typedef  itk::SigmoidImageFilter< FloatImageType, FloatImageType > SigmoidFilterType;
  SigmoidFilterType::Pointer sigmoid = SigmoidFilterType::New();
  sigmoid->SetOutputMinimum( 0.0 );
  sigmoid->SetOutputMaximum( 1.0 );
  sigmoid->SetAlpha( alpha );
  sigmoid->SetBeta( beta );
  sigmoid->SetInput( gradientMagnitude->GetOutput() );

  typedef  itk::FastMarchingImageFilter< FloatImageType, FloatImageType > FastMarchingFilterType;
  FastMarchingFilterType::Pointer  fastMarching = FastMarchingFilterType::New();

  typedef FastMarchingFilterType::NodeContainer  NodeContainer;
  typedef FastMarchingFilterType::NodeType       NodeType;

  FloatImageType::IndexType  seedPosition;
  seedPosition[ 0 ] = seedPosX;
  seedPosition[ 1 ] = seedPosY;

  NodeContainer::Pointer seeds = NodeContainer::New();
  NodeType node;
  node.SetValue( seedValue );
  node.SetIndex( seedPosition );

  seeds->Initialize();
  seeds->InsertElement( 0, node );

  fastMarching->SetTrialPoints( seeds );
  fastMarching->SetSpeedConstant( 1.0 );
  fastMarching->SetOutputSize( inputImage->GetBufferedRegion().GetSize() );

  typedef  itk::GeodesicActiveContourLevelSetImageFilter< FloatImageType, FloatImageType >  GeodesicActiveContourFilterType;
  GeodesicActiveContourFilterType::Pointer geodesicActiveContour = GeodesicActiveContourFilterType::New();
  geodesicActiveContour->SetPropagationScaling( propagationScaling );
  geodesicActiveContour->SetCurvatureScaling( 1.0 );
  geodesicActiveContour->SetAdvectionScaling( 1.0 );
  geodesicActiveContour->SetMaximumRMSError( 0.02 );
  geodesicActiveContour->SetNumberOfIterations( numberOfIterations );
  geodesicActiveContour->SetInput( fastMarching->GetOutput() );
  geodesicActiveContour->SetFeatureImage( sigmoid->GetOutput() );

  typedef itk::BinaryThresholdImageFilter< FloatImageType, UCImageType > ThresholdingFilterType;
  ThresholdingFilterType::Pointer thresholder = ThresholdingFilterType::New();
  thresholder->SetLowerThreshold( -1000.0 );
  thresholder->SetUpperThreshold( 0.0 );
  thresholder->SetOutsideValue( itk::NumericTraits< UCPixelType >::min() );
  thresholder->SetInsideValue( itk::NumericTraits< UCPixelType >::max() );
  thresholder->SetInput( geodesicActiveContour->GetOutput() );

  typedef itk::RescaleIntensityImageFilter< FloatImageType, UCImageType > CastFilterType;

  CastFilterType::Pointer caster1 = CastFilterType::New();
  CastFilterType::Pointer caster2 = CastFilterType::New();
  CastFilterType::Pointer caster3 = CastFilterType::New();
  CastFilterType::Pointer caster4 = CastFilterType::New();

  WriterType::Pointer writer1 = WriterType::New();
  caster1->SetInput( smoothing->GetOutput() );
  writer1->SetInput( caster1->GetOutput() );
  writer1->SetFileName( "Results\\GeodesicActiveContourImageFilterOutput1-smoothing.png" );
  caster1->SetOutputMinimum( itk::NumericTraits< UCPixelType >::min() );
  caster1->SetOutputMaximum( itk::NumericTraits< UCPixelType >::max() );
  writer1->Update();
  viewer.AddImage( caster1->GetOutput() );

  InternalWriterType::Pointer gradientWriter = InternalWriterType::New();
  gradientWriter->SetInput( gradientMagnitude->GetOutput() );
  gradientWriter->SetFileName( "Results\\GeodesicActiveContourImageFilterOutput2-gradientMagnitude.mha" );
  try
    {
    gradientWriter->Update();
    }
  catch( itk::ExceptionObject & error )
    {
    std::cerr << "Error: " << error << std::endl;
    return EXIT_FAILURE;
    }
  caster2->SetInput( gradientMagnitude->GetOutput() );
  caster2->SetOutputMinimum( itk::NumericTraits< UCPixelType >::min() );
  caster2->SetOutputMaximum( itk::NumericTraits< UCPixelType >::max() );
  viewer.AddImage( caster2->GetOutput() );

  InternalWriterType::Pointer speedWriter = InternalWriterType::New();
  speedWriter->SetInput( sigmoid->GetOutput() );
  speedWriter->SetFileName( "Results\\GeodesicActiveContourImageFilterOutput3-sigmoid.mha" );
  try
    {
    speedWriter->Update();
    }
  catch( itk::ExceptionObject & error )
    {
    std::cerr << "Error: " << error << std::endl;
    return EXIT_FAILURE;
    }
  caster3->SetInput( sigmoid->GetOutput() );
  caster3->SetOutputMinimum( itk::NumericTraits< UCPixelType >::min() );
  caster3->SetOutputMaximum( itk::NumericTraits< UCPixelType >::max() );
  viewer.AddImage( caster3->GetOutput() );

  InternalWriterType::Pointer mapWriter = InternalWriterType::New();
  mapWriter->SetInput( fastMarching->GetOutput() );
  mapWriter->SetFileName( "Results\\GeodesicActiveContourImageFilterOutput4-fastMarching.mha" );
  try
    {
    mapWriter->Update();
    }
  catch( itk::ExceptionObject & error )
    {
    std::cerr << "Error: " << error << std::endl;
    return EXIT_FAILURE;
    }
  caster4->SetInput( fastMarching->GetOutput() );
  caster4->SetOutputMinimum( itk::NumericTraits< UCPixelType >::min() );
  caster4->SetOutputMaximum( itk::NumericTraits< UCPixelType >::max() );
  viewer.AddImage( caster4->GetOutput() );


  viewer.AddImage( thresholder->GetOutput() );

  std::cout << std::endl;
  std::cout << "Max. no. iterations: " << geodesicActiveContour->GetNumberOfIterations() << std::endl;
  std::cout << "Max. RMS error: " << geodesicActiveContour->GetMaximumRMSError() << std::endl;
  std::cout << std::endl;
  std::cout << "No. elpased iterations: " << geodesicActiveContour->GetElapsedIterations() << std::endl;
  std::cout << "RMS change: " << geodesicActiveContour->GetRMSChange() << std::endl;

  viewer.Visualize();
  return EXIT_SUCCESS;
  }

int DirectEllipseFit( UCImageType::Pointer inputImage )
  {
  typedef itk::CastImageFilter< UCImageType, DoubleImageType >                      CastToDoubleFilterType;
  typedef itk::CannyEdgeDetectionImageFilter< DoubleImageType, DoubleImageType >    CannyFilterType;
  typedef itk::RescaleIntensityImageFilter< DoubleImageType, UCImageType >          RescaleFilterType;
  QuickView viewer;

  UCImageType::RegionType max_region = inputImage->GetLargestPossibleRegion();
  UCImageType::SizeType size = max_region.GetSize();
  //UCImageType::IndexType ind;
  //for( int i = 0; i < 5; i++ )
  //  {
  //  ind[ 0 ] = i;
  //  ind[ 1 ] = size[1]-i;
  //  inputImage->SetPixel( ind, 255 );
  //  }

  viewer.AddImage<UCImageType>( inputImage );

  // Edge detection (with smoothing)
  std::cout << "Edge detection" << std::endl;
  CastToDoubleFilterType::Pointer toDoubeFilter = CastToDoubleFilterType::New();
  toDoubeFilter->SetInput( inputImage );

  CannyFilterType::Pointer cannyFilter = CannyFilterType::New();
  cannyFilter->SetVariance( 45.0 );
  cannyFilter->SetLowerThreshold( 2.0 );
  cannyFilter->SetUpperThreshold( 5.0 );
  cannyFilter->SetInput( toDoubeFilter->GetOutput() );

  RescaleFilterType::Pointer rescaleFilter = RescaleFilterType::New();
  rescaleFilter->SetInput( cannyFilter->GetOutput() );
  rescaleFilter->SetOutputMinimum( 0 );
  rescaleFilter->SetOutputMaximum( 255 );

  typedef itk::StatisticsImageFilter<UCImageType> StatisticsImageFilterType;
  StatisticsImageFilterType::Pointer statisticsImageFilter
    = StatisticsImageFilterType::New();
  statisticsImageFilter->SetInput( rescaleFilter->GetOutput() );
  statisticsImageFilter->Update();

  std::cout << "Mean: " << statisticsImageFilter->GetMean() << std::endl;
  std::cout << "Std.: " << statisticsImageFilter->GetSigma() << std::endl;
  std::cout << "Min: " << int( statisticsImageFilter->GetMinimum() ) << std::endl;
  std::cout << "Max: " << int( statisticsImageFilter->GetMaximum() ) << std::endl;

  viewer.AddImage<UCImageType>( rescaleFilter->GetOutput() );

  // Direct Ellipse Fit Method
  typedef itk::VariableSizeMatrix<double> MatrixType;

  MatrixType x_coord, y_coord;
  x_coord.SetSize( (size[ 0 ] * size[ 1 ])/10, 1 );
  x_coord.Fill( 0 );
  y_coord.SetSize( (size[ 0 ] * size[ 1 ])/10, 1 );
  std::cout << "Initial size of x and y : " << size[ 0 ] * size[ 1 ] << std::endl;

  unsigned int non_zero_pixels = 0;

  double centroid_x = 0;
  double centroid_y = 0;
  UCImageType::IndexType index;

  for( unsigned int x = 0; x < size[ 0 ]; x++ )
    {
    for( unsigned int y = 0; y < size[ 1 ]; y++ )
      {  
      static unsigned int j = 0;
      index[ 0 ] = x;
      index[ 1 ] = y;
      if( rescaleFilter->GetOutput()->GetPixel( index ) == 255 )
        {
        x_coord( j, 0 ) = index[ 0 ];
        y_coord( j, 0 ) = index[ 1 ];
        std::cout << "x_coord = " << x_coord(j,0) << " y_coord = "<< y_coord(j,0) << std::endl;
        non_zero_pixels++;
        j++;
        centroid_x += index[ 0 ];
        centroid_y += index[ 1 ];
        }
      }
    }
  std::cout << "non_zero_pixels = " << non_zero_pixels << std::endl;
  centroid_x /= non_zero_pixels;
  centroid_y /= non_zero_pixels;
  std::cout << "centroid = [ " << centroid_x << ", " << centroid_y << " ]" << std::endl;

  //Method 1
/*MatrixType D;
  D.SetSize( non_zero_pixels, 6 );
  D.Fill( 0 );

  for( unsigned int i = 0; i < non_zero_pixels; i++ )
    {
    D( i, 0 ) = x_coord( i, 0 ) * x_coord( i, 0 );
    D( i, 1 ) = x_coord( i, 0 ) * y_coord( i, 0 );
    D( i, 2 ) = y_coord( i, 0 ) * y_coord( i, 0 );
    D( i, 3 ) = x_coord( i, 0 );
    D( i, 4 ) = y_coord( i, 0 );
    D( i, 5 ) = 1;
    }
  std::cout << "D : " << D[ non_zero_pixels - 1 ][ 0 ] << " " << D[ non_zero_pixels - 1 ][ 1 ] << " " << D[ non_zero_pixels - 1 ][ 2 ] << " " << D[ non_zero_pixels - 1 ][ 3 ] << " " << D[ non_zero_pixels - 1 ][ 4 ] << " " << D[ non_zero_pixels - 1 ][ 5 ] << std::endl;
  
  vnl_matrix<double> S = D.GetTranspose()*D.GetVnlMatrix();
  std::cout << "S = " << S << std::endl;

  MatrixType C;
  C.SetSize( 6, 6 );
  C.Fill( 0 );
  C( 0, 2 ) = 2;
  C( 1, 1 ) = -1;
  C( 2, 0 ) = 2;
  std::cout << "C = " << C << std::endl;

  typedef vnl_matrix< double > InputMatrixType;
  typedef itk::FixedArray< double, 6 > EigenValuesArrayType;
  typedef itk::Matrix< double, 6, 6 > EigenVectorMatrixType;
  typedef itk::SymmetricEigenAnalysis< InputMatrixType,
    EigenValuesArrayType, EigenVectorMatrixType > SymmetricEigenAnalysisType;

  InputMatrixType inputmatrix = vnl_matrix_inverse<double>( S ) * C.GetVnlMatrix();
  std::cout << "S'*C = " << inputmatrix << std::endl;

  EigenValuesArrayType eigenvalues;
  EigenVectorMatrixType eigenvectors;
  SymmetricEigenAnalysisType symmetricEigenSystem( 6 );
  symmetricEigenSystem.ComputeEigenValuesAndVectors( inputmatrix, eigenvalues, eigenvectors );

  std::cout << "EigenValues: " << eigenvalues << std::endl;
  std::cout << "EigenVectors (each row is an an eigen vector): " << std::endl;
  std::cout << eigenvectors << std::endl;


  double* eigvector;
  for( int pos = 0; pos < 6; pos++ )
    {
    if( eigenvalues[pos] > 0 )
      {
      eigvector = eigenvectors[ pos ];
      }
    }

  // We get an equation ax^2 + 2 bxy + cy^2 + 2dx + 2fy + g = 0
  std::cout << "eigenvector = " << *eigvector << std::endl;
  double a = eigvector[ 0 ];
  double b = eigvector[ 1 ];
  double c = eigvector[ 2 ];
  double d = eigvector[ 3 ];
  double f = eigvector[ 4 ];
  double g = eigvector[ 5 ];
*/

  //Method 2
  //MatrixType D1, D2;
  //D1.SetSize( non_zero_pixels, 3 );
  //D2.SetSize( non_zero_pixels, 3 );

  vnl_matrix<double> D1, D2;
  D1.set_size( non_zero_pixels, 3 );
  D2.set_size( non_zero_pixels, 3 );
  for( unsigned int j = 0; j < non_zero_pixels; j++ )
    {
    D1( j, 0 ) = pow( ( x_coord( j, 0 ) - centroid_x ), 2 );
    D1( j, 1 ) = ( x_coord( j, 0 ) - centroid_x )*( y_coord( j, 0 ) - centroid_y );
    D1( j, 2 ) = pow( ( y_coord( j, 0 ) - centroid_y ), 2 );
    D2( j, 0 ) = x_coord( j, 0 ) - centroid_x;
    D2( j, 1 ) = y_coord( j, 0 ) - centroid_y;
    D2( j, 2 ) = 1;
    }

  typedef vnl_matrix< double > VNLMatrixType;
  VNLMatrixType S1 = D1.transpose() * D1;
  VNLMatrixType S2 = D1.transpose() * D2;
  VNLMatrixType S3 = D2.transpose() * D2;
  std::cout << "S1 = " << S1 << std::endl;
  std::cout << "S2 = " << S2 << std::endl;
  std::cout << "S3 = " << S3 << std::endl;

  VNLMatrixType T = vnl_matrix_inverse<double>( S3 ) * S2.transpose() * (-1);
  std::cout << "T = " << T << std::endl;

  VNLMatrixType M1 = S1 + S2*T;
  std::cout << "M1 = " << M1 << std::endl;

  VNLMatrixType M2 = M1;
  M2.update( M1.get_n_rows( 2, 1 ) / 2, 0, 0 );
  M2.update( M1.get_n_rows( 1, 1 ) * (-1), 1, 0 );
  M2.update( M1.get_n_rows( 0, 1 ) / 2, 2, 0 );

  std::cout << "M2 = " << M2 << std::endl;

  typedef itk::FixedArray< double, 3 > EigenValuesArrayType;
  typedef itk::Matrix< double, 3, 3 > EigenVectorMatrixType;
  typedef itk::SymmetricEigenAnalysis< VNLMatrixType,
    EigenValuesArrayType, EigenVectorMatrixType > SymmetricEigenAnalysisType;

  EigenValuesArrayType eigenvalues;
  EigenVectorMatrixType eigenvectors;
  SymmetricEigenAnalysisType symmetricEigenSystem( 3 );
  symmetricEigenSystem.ComputeEigenValuesAndVectors( M2, eigenvalues, eigenvectors );
  std::cout << "EigenValues: " << eigenvalues << std::endl;
  std::cout << "EigenVectors (each row is an an eigen vector): " << std::endl;
  std::cout << eigenvectors << std::endl;

  vnl_vector<double> A1;
  bool cond_reached = false;
  for( int i = 0; i < eigenvalues.Length; i++ )
    {
    double cond = 4 * eigenvectors( i, 0 )*eigenvectors( i, 2 ) - pow( eigenvectors( i, 1 ), 2 );
    if( cond > 0 )
      {  
      A1 = eigenvectors.GetVnlMatrix().get_row( i );
      cond_reached = true;
      }
    }
  if( cond_reached == false )
    {
    std::cerr << "No eigenvector verify the condition." << std::endl;
    return EXIT_FAILURE;
    }
  std::cout << "A1 = " << A1 << std::endl;

  //VNLMatrixType A, A4, A5, A6;
  vnl_vector<double> A( 6 );
  double A3, A4, A5;
  A.update( A1, 0 );
  A( 3 ) = ( T*A1 )( 0 );
  A( 4 ) = ( T*A1 )( 1 );
  A( 5 ) = ( T*A1 )( 2 );

  std::cout << "T*A1 = " << T*A1 << std::endl;
  std::cout << "A = " << A << std::endl;

  A3 = A( 3 ) - 2 * A( 0 ) * centroid_x - A( 1 ) * centroid_y;
  A4 = A( 4 ) - 2 * A( 2 ) * centroid_y - A( 1 ) * centroid_x;
  A5 = A( 5 ) + A( 0 ) * pow( centroid_x, 2 ) + A( 2 ) * pow( centroid_y, 2 ) + A( 1 ) * centroid_x*centroid_y - A( 3 ) * centroid_x - A( 4 ) * centroid_y;

  std::cout << "A3 = " << A3 << " A4 = " << A4 << " A5 = " << A5 << std::endl;

  A( 3 ) = A3;
  A( 4 ) = A4;
  A( 5 ) = A5;

  std::cout << "A = " << A << std::endl;
  A.normalize();
  std::cout << "A = " << A << std::endl;

  double a = A( 0 );
  double b = A( 1 ) / 2;
  double c = A( 2 );
  double d = A( 3 ) / 2;
  double f = A( 4 ) / 2;
  double g = A( 5 );


  // define the center of the ellipse
  RGBImageType::IndexType center;
  center[ 0 ] = ( c*d - b*f ) / ( b*b - a*c );
  center[ 1 ] = ( a*f - b*d ) / ( b*b - a*c );
  std::cout << "center = " << center << std::endl;

  // semi-axis lengths
  double inter = std::sqrt( pow( ( a - c ), 2 ) + 4 * b*b );
  double num = 2 * ( a*f*f + c*d*d + g*b*b - 2 * b*d*f - a*c*g );
  double minor_axis = std::sqrt( num / ( ( b*b - a*c )*( inter - ( a + c ) ) ) );
  double major_axis = std::sqrt( num / ( ( b*b - a*c )*( inter * (-1) - ( a + c ) ) ) );
  std::cout << "inter = " << inter << std::endl;
  std::cout << "num = " << num << std::endl;
  std::cout << "semi-axis lengths = " << minor_axis << " " << major_axis << std::endl;
  if( minor_axis == NAN || major_axis == NAN || minor_axis == 0 || major_axis == 0 )
    {
    std::cerr << "Error in the computation of the length of the axis" << std::endl;
    return EXIT_FAILURE;
    }


  // counterclockwise angle of rotation from the x-axis to the major axis 
  double alpha = 9999;
  if( b == 0 )
    {
    if( a < c )
      {
      alpha = 0;
      }
    else
      {
      alpha = 0.5 * M_PI;
      }
    }
  else
    {
    double cotan = cos( ( a - c ) / 2 * b ) / sin( ( a - c ) / 2 * b );
    if( a < c )
      {
      alpha = 1 / ( 2 * cotan );
      }
    else
      {
      alpha = M_PI + 1 / ( 2 * cotan );
      }
    }
  std::cout << "alpha = " << alpha << std::endl;
  if( alpha == 9999 )
    {
    std::cerr << "Error with the computation of alpha." << std::endl;
    return EXIT_FAILURE;
    }

  /*// draw the minor and major axis 
  CastToRGBFilterType::Pointer toRGBFilter = CastToRGBFilterType::New();
  toRGBFilter->SetInput( inputImage );
  toRGBFilter->Update();
  RGBImageType::Pointer image = toRGBFilter->GetOutput();

  RGBImageType::PixelType pixel;
  pixel.SetRed( 255 );
  pixel.SetGreen( 0 );
  pixel.SetBlue( 0 );

  RGBImageType::IndexType localIndex;
  for( double r = -major_axis; r <= major_axis; r+=10 )
    {
    localIndex[ 0 ] = center[ 0 ] + r*cos( alpha );
    localIndex[ 1 ] = center[ 1 ] + r*sin( alpha );
    image->SetPixel( localIndex, pixel );
    }
  for( double r = -minor_axis; r <= minor_axis; r+=10 )
    {
    localIndex[ 0 ] = center[ 0 ] + r*cos( alpha + M_PI/2 );
    localIndex[ 1 ] = center[ 1 ] + r*sin( alpha + M_PI/2 );
    image->SetPixel( localIndex, pixel );
    }
  viewer.AddRGBImage<RGBImageType>( image );*/

  typedef itk::EllipseSpatialObject< 2 >   EllipseType;
  typedef itk::SpatialObjectToImageFilter<
    EllipseType, UCImageType >   SpatialObjectToImageFilterType;

  SpatialObjectToImageFilterType::Pointer imageFilter =
    SpatialObjectToImageFilterType::New();

  imageFilter->SetSize( size );
  const UCImageType::SpacingType& spacing = inputImage->GetSpacing();
  imageFilter->SetSpacing( spacing );
  const UCImageType::PointType& origin = inputImage->GetOrigin();
  imageFilter->SetOrigin( origin );

  EllipseType::Pointer ellipse = EllipseType::New();
  EllipseType::ArrayType radiusArray;
  radiusArray[ 0 ] = minor_axis;
  radiusArray[ 1 ] = major_axis;
  ellipse->SetRadius( radiusArray );

  typedef EllipseType::TransformType TransformType;
  TransformType::Pointer transform = TransformType::New();

  const double imageCenterX = origin[ 0 ] + spacing[ 0 ] * size[ 0 ] / 2;
  const double imageCenterY = origin[ 1 ] + spacing[ 1 ] * size[ 1 ] / 2;

  //TransformType::OutputVectorType  translation1;
  //translation1[ 0 ] = -imageCenterX;
  //translation1[ 1 ] = -imageCenterY;
  //transform->Translate( translation1, false );
  
  transform->Rotate2D( -M_PI/4 );

  TransformType::OutputVectorType  translation2;
  translation2[ 0 ] = center[ 0 ];// imageCenterX + center[ 0 ];
  translation2[ 1 ] = center[ 1 ];// imageCenterY + center[ 1 ];
  transform->Translate( translation2 );

  ellipse->SetObjectToParentTransform( transform );

  imageFilter->SetInput( ellipse );
  ellipse->SetDefaultInsideValue( 255 );
  ellipse->SetDefaultOutsideValue( 0 );
  imageFilter->SetUseObjectValue( true );
  imageFilter->SetOutsideValue( 0 );

  try
    {
    imageFilter->Update();
    }
  catch( itk::ExceptionObject & excp )
    {
    std::cerr << excp << std::endl;
    return EXIT_FAILURE;
    }
  viewer.AddImage( imageFilter->GetOutput() );


  viewer.Visualize();

  return EXIT_SUCCESS;
  }

int SpatialObjectImageRegistrationMethod( UCImageType::Pointer inputImage )
  {
  QuickView viewer;
  viewer.AddImage<UCImageType>( inputImage, false, "inputImage" );
  UCImageType::RegionType max_region = inputImage->GetLargestPossibleRegion();
  UCImageType::SizeType size_input = max_region.GetSize();

  // Prepare the input image : convert into float and blur it
  typedef itk::CastImageFilter< UCImageType, FloatImageType >                CastToFloatFilterType;

  CastToFloatFilterType::Pointer toFloatFilter = CastToFloatFilterType::New();
  toFloatFilter->SetInput( inputImage );

  /*typedef itk::CannyEdgeDetectionImageFilter< FloatImageType, FloatImageType >
    CannyFilterType;

  CannyFilterType::Pointer cannyFilter = CannyFilterType::New();
  cannyFilter->SetVariance( 45.0 );
  cannyFilter->SetLowerThreshold( 2.0 );
  cannyFilter->SetUpperThreshold( 5.0 );
  cannyFilter->SetInput( toFloatFilter->GetOutput() );*/

  typedef itk::DiscreteGaussianImageFilter<FloatImageType, FloatImageType >    DiscretGaussianFilterType;
  DiscretGaussianFilterType::Pointer gaussianFilter = DiscretGaussianFilterType::New();
  float variance = 20.0;
  gaussianFilter->SetInput( toFloatFilter->GetOutput() );
  gaussianFilter->SetVariance( variance );
  gaussianFilter->Update();

  typedef itk::Image<float, 2> FixedImageType;
  FixedImageType::Pointer fixedImage = FixedImageType::New();
  fixedImage = gaussianFilter->GetOutput();

  viewer.AddImage<FloatImageType>( fixedImage, false, "Blurred input image" );


  // Definition of the ellipse
  typedef itk::EllipseSpatialObject< 2 >   EllipseType;

  EllipseType::Pointer ellipse = EllipseType::New();
  EllipseType::ArrayType radius;
  radius[ 0 ] = 100;
  radius[ 1 ] = 40;
  ellipse->SetRadius( radius );

  EllipseType::TransformType::OffsetType offset;
  offset[ 0 ] = size_input[ 0 ] / 2;
  offset[ 1 ] = size_input[ 1 ] / 2;
  ellipse->GetObjectToParentTransform()->SetOffset( offset );
  ellipse->ComputeObjectToWorldTransform();

  ellipse->SetDefaultInsideValue( 255 );
  ellipse->SetDefaultOutsideValue( 0 );

  // View the original ellipse
  typedef itk::SpatialObjectToImageFilter< EllipseType, FixedImageType >
    SpatialObjectToImageFilterType;
  SpatialObjectToImageFilterType::Pointer imageFilter =
    SpatialObjectToImageFilterType::New();
  imageFilter->SetInput( ellipse );
  imageFilter->SetSize( size_input );
  imageFilter->Update();

  imageFilter->SetUseObjectValue( true );
  imageFilter->SetOutsideValue( 0 );

  typedef itk::RescaleIntensityImageFilter< FloatImageType, UCImageType >    RescaleFilterType;
  RescaleFilterType::Pointer rescaleFilter = RescaleFilterType::New();
  rescaleFilter->SetInput( imageFilter->GetOutput() );
  rescaleFilter->SetOutputMinimum( 0 );
  rescaleFilter->SetOutputMaximum( 255 );

  viewer.AddImage<UCImageType>( rescaleFilter->GetOutput(), false, "Original ellipse" );

  /*typedef itk::ImageFileWriter< UCImageType >     WriterType;
  WriterType::Pointer writer = WriterType::New();
  writer->SetFileName( "C:\\MyProjects\\Results\\ellipse.png" );
  writer->SetInput( rescaleFilter->GetOutput() );
  try
    {
    imageFilter->Update();
    writer->Update();
    }
  catch( itk::ExceptionObject & excp )
    {
    std::cerr << excp << std::endl;
    return EXIT_FAILURE;
    }

  typedef itk::DiscreteGaussianImageFilter< FloatImageType, FloatImageType >
    GaussianFilterType;
  GaussianFilterType::Pointer   gaussianFilter = GaussianFilterType::New();
  gaussianFilter->SetInput( imageFilter->GetOutput() );
  const double variance = 20;
  gaussianFilter->SetVariance( variance );
  gaussianFilter->Update();*/

  // Definition of the registration
  typedef itk::ImageToSpatialObjectRegistrationMethod< FloatImageType, EllipseType >
    RegistrationType;
  RegistrationType::Pointer registration = RegistrationType::New();

  typedef itk::SimpleImageToSpatialObjectMetric< FloatImageType, EllipseType > MetricType;
  MetricType::Pointer metric = MetricType::New();

  typedef itk::LinearInterpolateImageFunction< FloatImageType, double >
    InterpolatorType;
  InterpolatorType::Pointer interpolator = InterpolatorType::New();

  typedef itk::OnePlusOneEvolutionaryOptimizer  OptimizerType;
  OptimizerType::Pointer optimizer = OptimizerType::New();

  typedef itk::CenteredSimilarity2DTransform<double> TransformType;
  TransformType::Pointer transform = TransformType::New();

  itk::Statistics::NormalVariateGenerator::Pointer generator
    = itk::Statistics::NormalVariateGenerator::New();
  generator->Initialize( 12345 );

  optimizer->SetNormalVariateGenerator( generator );
  optimizer->Initialize( 10 );
  optimizer->SetMaximumIteration( 500 );

  /*typedef itk::CenteredTransformInitializer<
    TransformType,
    FixedImageType,
    EllipseType > TransformInitializerType;
  TransformInitializerType::Pointer initializer = TransformInitializerType::New();
  initializer->SetTransform( transform );
  initializer->SetFixedImage( fixedImage );
  initializer->SetMovingImage( ellipse );
  initializer->MomentsOn();
  initializer->InitializeTransform();*/

  registration->SetInitialTransformParameters( transform->GetParameters() );
  
  TransformType::ParametersType initialParameters(
    transform->GetNumberOfParameters() );
  initialParameters[ 0 ] = 1.0;     // Scale
  initialParameters[ 1 ] = 0.0;     // Angle
  initialParameters[ 2 ] = 317.5;   // Rotation center X
  initialParameters[ 3 ] = 457.5;   // Rotation center Y
  initialParameters[ 4 ] = 0.0;     // Translation X
  initialParameters[ 5 ] = 0.0;     // Translation Y
  registration->SetInitialTransformParameters( initialParameters );
  std::cout << "Initial Parameters  : " << initialParameters << std::endl;

  // View the ellipse after the initial transform
  SpatialObjectToImageFilterType::Pointer imageFilter2 =
    SpatialObjectToImageFilterType::New();
  imageFilter2->SetInput( ellipse );
  imageFilter2->SetSize( size_input );
  imageFilter2->Update();

  imageFilter2->SetUseObjectValue( true );
  imageFilter2->SetOutsideValue( 0 );

  typedef itk::ResampleImageFilter< FixedImageType,
    FixedImageType > ResampleFilterType2;
  ResampleFilterType2::Pointer resampler2 = ResampleFilterType2::New();
  resampler2->SetTransform( transform );
  resampler2->SetInput( imageFilter2->GetOutput() );
  resampler2->SetSize( fixedImage->GetLargestPossibleRegion().GetSize() );
  resampler2->SetOutputOrigin( fixedImage->GetOrigin() );
  resampler2->SetOutputSpacing( fixedImage->GetSpacing() );
  resampler2->SetOutputDirection( fixedImage->GetDirection() );
  resampler2->SetDefaultPixelValue( 100 );
  viewer.AddImage<FixedImageType>( resampler2->GetOutput(), false, "movingImage after the initial tranform" );


  /*TransformType::ParametersType parametersScale;
  parametersScale.set_size( 3 );
  parametersScale[ 0 ] = 1000; // angle scale
  for( unsigned int i = 1; i < 3; i++ )
    {
    parametersScale[ i ] = 2; // offset scale
    }
  optimizer->SetScales( parametersScale );*/
  typedef OptimizerType::ScalesType       OptimizerScalesType;
  OptimizerScalesType optimizerScales( transform->GetNumberOfParameters() );
  const double translationScale = 1.0 / 100.0;
  optimizerScales[ 0 ] = 10.0;
  optimizerScales[ 1 ] = 1.0;
  optimizerScales[ 2 ] = translationScale;
  optimizerScales[ 3 ] = translationScale;
  optimizerScales[ 4 ] = translationScale;
  optimizerScales[ 5 ] = translationScale;
  optimizer->SetScales( optimizerScales );

  typedef itk::IterationCallback< OptimizerType >   IterationCallbackType;
  IterationCallbackType::Pointer callback = IterationCallbackType::New();
  callback->SetOptimizer( optimizer );

  registration->SetFixedImage( fixedImage );
  registration->SetMovingSpatialObject( ellipse );
  registration->SetTransform( transform );
  registration->SetInterpolator( interpolator );
  registration->SetOptimizer( optimizer );
  registration->SetMetric( metric );

  optimizer->MaximizeOn();

  try
    {
    registration->Update();
    std::cout << "Optimizer stop condition: "
      << registration->GetOptimizer()->GetStopConditionDescription()
      << std::endl;
    }
  catch( itk::ExceptionObject & exp )
    {
    std::cerr << "Exception caught ! " << std::endl;
    std::cerr << exp << std::endl;
    }

  RegistrationType::ParametersType finalParameters
    = registration->GetLastTransformParameters();
  TransformType::ParametersType param = transform->GetParameters();
  std::cout << "Final Solution is : " << finalParameters << std::endl;
  std::cout << " Transform parameters = " << param << std::endl;

  TransformType::ParametersType finalParametersTransform =
    transform->GetParameters();
  const double finalScale = finalParametersTransform[ 0 ];
  const double finalAngle = finalParametersTransform[ 1 ];
  const double finalRotationCenterX = finalParametersTransform[ 2 ];
  const double finalRotationCenterY = finalParametersTransform[ 3 ];
  const double finalTranslationX = finalParametersTransform[ 4 ];
  const double finalTranslationY = finalParametersTransform[ 5 ];
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

  // Apply the final transform to the ellipse
  SpatialObjectToImageFilterType::Pointer imageFilter3 =
    SpatialObjectToImageFilterType::New();
  imageFilter3->SetInput( ellipse );
  imageFilter3->SetSize( inputImage->GetLargestPossibleRegion().GetSize() );
  imageFilter3->Update();
  imageFilter3->SetUseObjectValue( true );
  imageFilter3->SetOutsideValue( 0 );

  typedef itk::ResampleImageFilter< FixedImageType,
    FixedImageType > ResampleFilterType;
  ResampleFilterType::Pointer resampler = ResampleFilterType::New();
  resampler->SetTransform( transform );
  resampler->SetInput( imageFilter3->GetOutput() );
  resampler->SetSize( inputImage->GetLargestPossibleRegion().GetSize() );
  resampler->SetOutputOrigin( inputImage->GetOrigin() );
  resampler->SetOutputSpacing( inputImage->GetSpacing() );
  resampler->SetOutputDirection( inputImage->GetDirection() );
  resampler->SetDefaultPixelValue( 100 );

  viewer.AddImage<FloatImageType>( resampler->GetOutput(), false, "Ellipse after registration" );

  viewer.Visualize();
  return EXIT_SUCCESS;
  }

int SimilarityTransform2DMethod( UCImageType::Pointer inputImage, unsigned int number, int threshold )
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

  // Smooth while preserving the edges
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
  //reader->SetFileName( "C:\\MyProjects\\018_crop.png" );
  //reader->SetFileName( "C:\\MyProjects\\ellipse1.png" );
  unsigned int number = 0;
  for( number = 16; number < 17; number++ )
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

    //HoughMethod( reader->GetOutput() );
    //int success = ParametricSnakeMethod( reader->GetOutput() );
    //int success = DirectEllipseFit( reader->GetOutput() );
    //int success = SpatialObjectImageRegistrationMethod( reader->GetOutput() );
    int success = SimilarityTransform2DMethod( reader->GetOutput(), number, threshold );
    if( success != EXIT_SUCCESS )
      {
      std::cerr << "Error in the selected method" << std::endl;
      }
    }

  return 0;
  }

