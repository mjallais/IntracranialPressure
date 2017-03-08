
#include "HoughMethod.cpp"

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


#include "QuickView.h"

#include <iostream>

  typedef unsigned char                                                   UCPixelType;
  typedef itk::RGBPixel< UCPixelType >                                    RGBPixelType;
  typedef itk::Image< UCPixelType, 2>                                     UCImageType;
  typedef itk::Image< double, 2 >                                         DoubleImageType;
  typedef itk::Image< float, 2 >                                          FloatImageType;
  typedef itk::Image< RGBPixelType, 2 >                                   RGBImageType;


void HoughMethod( UCImageType::Pointer inputImage )
{
  typedef float                                                           AccumulatorPixelType;
  typedef itk::DiscreteGaussianImageFilter<UCImageType, UCImageType >         DiscretGaussianFilterType;
  typedef itk::CastImageFilter< UCImageType, DoubleImageType >              CastToDoubleFilterType;
  typedef itk::CannyEdgeDetectionImageFilter< DoubleImageType, DoubleImageType >
    CannyFilterType;
  typedef itk::RescaleIntensityImageFilter< DoubleImageType, UCImageType >  RescaleFilterType;
  typedef itk::HoughTransform2DCirclesImageFilter<UCPixelType, AccumulatorPixelType>
    HoughTransformFilterType;
  typedef itk::Image< AccumulatorPixelType, 2 >                           AccumulatorImageType;

  typedef itk::CastImageFilter< UCImageType, RGBImageType>                  CastToRGBFilterType;
  
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

int ParametricSnakeMethod ( UCImageType::Pointer inputImage )
{ 
  typedef  itk::ImageFileWriter< UCImageType >                        WriterType;
  typedef itk::ImageFileWriter< FloatImageType >                      InternalWriterType;

  QuickView viewer;

  UCImageType::RegionType max_region = inputImage->GetLargestPossibleRegion();
  UCImageType::SizeType size = max_region.GetSize();
  std::cout << "inputImage size = " << size << std::endl;

  viewer.AddImage( inputImage.GetPointer() );

  const int seedPosX = size[0]/2;
  const int seedPosY = size[1]/2;

  const double initialDistance = 5.0;
  const double sigma = 1.0;
  const double alpha = -0.5;
  const double beta = 3.0;
  const double propagationScaling = 2.0;
  const double numberOfIterations = 5.0;
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

int main()
  {
  typedef itk::ImageFileReader<UCImageType>                               ImageReaderType;

  // Read the image
  std::cout << "Read the image" << std::endl;
  ImageReaderType::Pointer reader = ImageReaderType::New();
  reader->SetFileName( "C:\\IntracranialPressure\\ONSD-Data\\018.png" );
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

  //HoughMethod( reader->GetOutput() );
  int success = ParametricSnakeMethod( reader->GetOutput() );
  if( success != EXIT_SUCCESS )
    {
    std::cerr << "Error in ParametricSnakeMethod" << std::endl;
    }

  return 0;
  }