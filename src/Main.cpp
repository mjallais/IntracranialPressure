
#include "itkCannyEdgeDetectionImageFilter.h"
#include "itkCastImageFilter.h"
#include "itkDiscreteGaussianImageFilter.h"
#include "itkHoughTransform2DCirclesImageFilter.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkStatisticsImageFilter.h"

#include "QuickView.h"

#include <iostream>

int main ()
{
  typedef unsigned char                                                   PixelType;
  typedef float                                                           AccumulatorPixelType;
  typedef itk::Image< PixelType, 2>                                       ImageType;
  typedef itk::Image< double, 2 >                                         DoubleImageType;
  typedef itk::ImageFileReader<ImageType>                                 ImageReaderType;
  typedef itk::DiscreteGaussianImageFilter<ImageType, ImageType >         DiscretGaussianFilterType;
  typedef itk::StatisticsImageFilter< DoubleImageType >                   StatisticsFilterType;
  typedef itk::CastImageFilter< ImageType, DoubleImageType >              CastToDoubleFilterType;
  typedef itk::CannyEdgeDetectionImageFilter< DoubleImageType, DoubleImageType > 
    CannyFilterType;
  typedef itk::RescaleIntensityImageFilter< DoubleImageType, ImageType >  RescaleFilterType;
  typedef itk::HoughTransform2DCirclesImageFilter<unsigned char, AccumulatorPixelType> 
    HoughTransformFilterType;
  typedef itk::Image< AccumulatorPixelType, 2 >                           AccumulatorImageType;

  // 1 : Read the image
  ImageReaderType::Pointer reader = ImageReaderType::New();
  reader->SetFileName( "C:\\IntracranialPressure\\ONSD-Data\\010.png" );
  
  QuickView viewer;
  viewer.AddImage<ImageType>( reader->GetOutput() );

  ImageType::RegionType max_region = reader->GetOutput()->GetLargestPossibleRegion();
  ImageType::SizeType size = max_region.GetSize();
  std::cout << " size = " << size << std::endl;

  // 2 : Smoothing
  float variance = 5.0;

  DiscretGaussianFilterType::Pointer gaussianFilter = DiscretGaussianFilterType::New();
  gaussianFilter->SetInput( reader->GetOutput() );
  gaussianFilter->SetVariance( variance );
  viewer.AddImage<ImageType>( gaussianFilter->GetOutput() );

  // 3 : Edge Detection
  CastToDoubleFilterType::Pointer toDoubeFilter = CastToDoubleFilterType::New();
  toDoubeFilter->SetInput( gaussianFilter->GetOutput() );

  StatisticsFilterType::Pointer statisticsFilter = StatisticsFilterType::New();
  statisticsFilter->SetInput( toDoubeFilter->GetOutput() );
  statisticsFilter->Update();
  double mean = statisticsFilter->GetMean();
  std::cout << "mean = " << mean << std::endl;
  std::cout << "min = " << double(statisticsFilter->GetMinimum()) << std::endl;
  std::cout << "max = " << double(statisticsFilter->GetMaximum()) << std::endl;

  float sigma = 0.33;
  float upperThreshold = std::min( 255.0 , (1.0 + sigma ) * mean );
  float lowerThreshold = std::max( 0.0, ( 1.0 - sigma ) * mean );
  std::cout << "upperthreshold = " << upperThreshold << std::endl;
  std::cout << "lowerthreshold = " << lowerThreshold << std::endl;

  CannyFilterType::Pointer cannyFilter = CannyFilterType::New();
  cannyFilter->SetVariance( 20.0 );
  cannyFilter->SetLowerThreshold( 0.0 );
  cannyFilter->SetUpperThreshold( 4.0 );

  cannyFilter->SetInput( toDoubeFilter->GetOutput() );
  std::cout << "upperthreshold = " << cannyFilter->GetUpperThreshold() << std::endl;
  std::cout << "lowerthreshold = " << cannyFilter->GetLowerThreshold() << std::endl;
  std::cout << "variance = " << cannyFilter->GetVariance() << std::endl;

  RescaleFilterType::Pointer rescaleFilter = RescaleFilterType::New();
  rescaleFilter->SetInput( cannyFilter->GetOutput() );
  rescaleFilter->SetOutputMinimum( 0 );
  rescaleFilter->SetOutputMaximum( 255 );
  viewer.AddImage<ImageType>( rescaleFilter->GetOutput() );

  // 4 : Hough Transform
  HoughTransformFilterType::Pointer houghFilter = HoughTransformFilterType::New();
  houghFilter->SetInput( rescaleFilter->GetOutput() );

  houghFilter->SetNumberOfCircles( 1 );
  houghFilter->SetMinimumRadius( 5 );
  houghFilter->SetMaximumRadius( size[0]/3 );
  houghFilter->SetSweepAngle( 100 );
  houghFilter->SetSigmaGradient(2);
  houghFilter->Update();

  AccumulatorImageType::Pointer localAccumulator = houghFilter->GetOutput();
  HoughTransformFilterType::CirclesListType circles;
  circles = houghFilter->GetCircles( 1 );

  ImageType::Pointer localOutputImage = ImageType::New();
  ImageType::RegionType region;
  region.SetSize( rescaleFilter->GetOutput()->GetLargestPossibleRegion().GetSize() );
  region.SetIndex( rescaleFilter->GetOutput()->GetLargestPossibleRegion().GetIndex() );
  localOutputImage->SetRegions( region );
  localOutputImage->SetOrigin( rescaleFilter->GetOutput()->GetOrigin() );
  localOutputImage->SetSpacing( rescaleFilter->GetOutput()->GetSpacing() );
  localOutputImage->Allocate();
  localOutputImage->FillBuffer( 0 );

  ImageType::IndexType localIndex;
  typedef HoughTransformFilterType::CirclesListType CirclesListType;
  CirclesListType::const_iterator itCircles = circles.begin();
  while( itCircles != circles.end() )
    {
    for( double angle = 0; angle <= 2 * vnl_math::pi; angle += vnl_math::pi / 60.0 )
      {
      localIndex[ 0 ] =
        ( long int )( ( *itCircles )->GetObjectToParentTransform()->GetOffset()[ 0 ]
          + ( *itCircles )->GetRadius()[ 0 ] * vcl_cos( angle ) );
      localIndex[ 1 ] =
        ( long int )( ( *itCircles )->GetObjectToParentTransform()->GetOffset()[ 1 ]
          + ( *itCircles )->GetRadius()[ 0 ] * vcl_sin( angle ) );
      ImageType::RegionType outputRegion =
        localOutputImage->GetLargestPossibleRegion();
      if( outputRegion.IsInside( localIndex ) )
        {
        localOutputImage->SetPixel( localIndex, 255 );
        }
      }
    itCircles++;
    }

  viewer.AddImage<ImageType>( localOutputImage );

  viewer.Visualize();

  return EXIT_SUCCESS;
}