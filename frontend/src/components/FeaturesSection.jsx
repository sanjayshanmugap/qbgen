import { features } from "../data/features";

const FeaturesSection = () => {
    return (
        <div className="flex items-center p-10 justify-center flex-col">
            <h2 className="font-bold text-3xl md:text-5xl mb-4 pt-20">Quiz Bowl AI Tools</h2>
            <div className="mt-10 grid items-center grid-cols-1 gap-10 md:grid-cols-3 max-w-screen-xl">
                {features.map((feature, index) => (
                    <div key={index} className="bg-white border border-indigo-400/30 rounded-lg shadow-lg p-6 h-full flex space-x-4">
                        <div>
                            <h3 className="font-semibold text-xl">{feature.title}</h3>
                            <p className="text-gray-800">{feature.description}</p>
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
};


export default FeaturesSection;