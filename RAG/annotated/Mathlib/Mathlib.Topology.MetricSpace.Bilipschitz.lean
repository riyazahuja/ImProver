/-- If `f : α → β` is bilipschitz, then the pullback of the uniformity on `β` through `f` agrees
with the uniformity on `α`.

This can be used to provide the replacement equality when applying
`PseudoMetricSpace.replaceUniformity`, which can be useful when following the forgetful inheritance
pattern when creating type synonyms.

Important Note: if `α` is some synonym of a type `β` (at default transparency), and `f : α ≃ β` is
some bilipschitz equivalence, then instead of writing:
```
instance : UniformSpace α := inferInstanceAs (UniformSpace β)
```
Users should instead write something like:
```
instance : UniformSpace α := (inferInstance : UniformSpace β).comap f
```
in order to avoid abuse of the definitional equality `α := β`. -/
lemma uniformity_eq_of_bilipschitz (hf₁ : AntilipschitzWith K₁ f) (hf₂ : LipschitzWith K₂ f) :
    𝓤[(inferInstance : UniformSpace β).comap f] = 𝓤 α :=
  hf₁.isUniformInducing hf₂.uniformContinuous |>.comap_uniformity


/-- If `f : α → β` is bilipschitz, then the pullback of the bornology on `β` through `f` agrees
with the bornology on `α`. -/
lemma bornology_eq_of_bilipschitz (hf₁ : AntilipschitzWith K₁ f) (hf₂ : LipschitzWith K₂ f) :
    @cobounded _ (induced f) = cobounded α :=
  le_antisymm hf₂.comap_cobounded_le hf₁.tendsto_cobounded.le_comap



/-- If `f : α → β` is bilipschitz, then the pullback of the bornology on `β` through `f` agrees
with the bornology on `α`.

This can be used to provide the replacement equality when applying
`PseudoMetricSpace.replaceBornology`, which can be useful when following the forgetful inheritance
pattern when creating type synonyms.

Important Note: if `α` is some synonym of a type `β` (at default transparency), and `f : α ≃ β` is
some bilipschitz equivalence, then instead of writing:
```
instance : Bornology α := inferInstanceAs (Bornology β)
```
Users should instead write something like:
```
instance : Bornology α := Bornology.induced (f : α → β)
```
in order to avoid abuse of the definitional equality `α := β`. -/
lemma isBounded_iff_of_bilipschitz (hf₁ : AntilipschitzWith K₁ f) (hf₂ : LipschitzWith K₂ f)
    (s : Set α) : @IsBounded _ (induced f) s ↔ Bornology.IsBounded s :=
  Filter.ext_iff.1 (bornology_eq_of_bilipschitz hf₁ hf₂) (sᶜ)


