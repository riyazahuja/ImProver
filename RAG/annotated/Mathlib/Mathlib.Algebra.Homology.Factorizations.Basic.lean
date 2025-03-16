/-- A morphism of cochain complexes `φ` in an abelian category satisfies
`degreewiseEpiWithInjectiveKernel φ` if for any `i : ℤ`, the morphism
`φ.f i` is an epimorphism with an injective kernel. -/
def degreewiseEpiWithInjectiveKernel : MorphismProperty (CochainComplex C ℤ) :=
  fun _ _ φ => ∀ (i : ℤ), epiWithInjectiveKernel (φ.f i)


instance : (degreewiseEpiWithInjectiveKernel (C := C)).IsMultiplicative where
  id_mem _ _ := MorphismProperty.id_mem _ _
  comp_mem _ _ hf hg n := MorphismProperty.comp_mem _ _ _ (hf n) (hg n)


