@[to_additive]
theorem equicontinuous_of_equicontinuousAt_one {ι G M hom : Type*} [TopologicalSpace G]
    [UniformSpace M] [Group G] [Group M] [TopologicalGroup G] [UniformGroup M]
    [FunLike hom G M] [MonoidHomClass hom G M] (F : ι → hom)
    (hf : EquicontinuousAt ((↑) ∘ F) (1 : G)) :
    Equicontinuous ((↑) ∘ F) := by
  /-
    ι : Type u_1
    G : Type u_2
    M : Type u_3
    hom : Type u_4
    inst✝⁷ : TopologicalSpace G
    inst✝⁶ : UniformSpace M
    inst✝⁵ : Group G
    inst✝⁴ : Group M
    inst✝³ : TopologicalGroup G
    inst✝² : UniformGroup M
    inst✝¹ : FunLike hom G M
    inst✝ : MonoidHomClass hom G M
    F : ι → hom
    hf : EquicontinuousAt (Function.comp DFunLike.coe F) 1
    ⊢ Equicontinuous (Function.comp DFunLike.coe F)
  -/
  rw [equicontinuous_iff_continuous]
  /-
    ι : Type u_1
    G : Type u_2
    M : Type u_3
    hom : Type u_4
    inst✝⁷ : TopologicalSpace G
    inst✝⁶ : UniformSpace M
    inst✝⁵ : Group G
    inst✝⁴ : Group M
    inst✝³ : TopologicalGroup G
    inst✝² : UniformGroup M
    inst✝¹ : FunLike hom G M
    inst✝ : MonoidHomClass hom G M
    F : ι → hom
    hf : EquicontinuousAt (Function.comp DFunLike.coe F) 1
    ⊢ Continuous (Function.comp (⇑UniformFun.ofFun) (Function.swap (Function.comp  …
  -/
  rw [equicontinuousAt_iff_continuousAt] at hf
  let φ : G →* (ι →ᵤ M) :=
    { toFun := swap ((↑) ∘ F)
      map_one' := by dsimp [UniformFun]; ext; exact map_one _
      map_mul' := fun a b => by dsimp [UniformFun]; ext; exact map_mul _ _ _ }
  /-
    ι : Type u_1
    G : Type u_2
    M : Type u_3
    hom : Type u_4
    inst✝⁷ : TopologicalSpace G
    inst✝⁶ : UniformSpace M
    inst✝⁵ : Group G
    inst✝⁴ : Group M
    inst✝³ : TopologicalGroup G
    inst✝² : UniformGroup M
    inst✝¹ : FunLike hom G M
    inst✝ : MonoidHomClass hom G M
    F : ι → hom
    hf : ContinuousAt (Function.comp (⇑UniformFun.ofFun) (Function.swap (Function. …
    φ : MonoidHom G (UniformFun ι M) := { toFun := Function.swap (Function.comp DF …
    ⊢ Continuous (Function.comp (⇑UniformFun.ofFun) (Function.swap (Function.comp  …
  -/
  exact continuous_of_continuousAt_one φ hf
  /-
    🎉 no goals
  -/


@[to_additive]
theorem uniformEquicontinuous_of_equicontinuousAt_one {ι G M hom : Type*} [UniformSpace G]
    [UniformSpace M] [Group G] [Group M] [UniformGroup G] [UniformGroup M]
    [FunLike hom G M] [MonoidHomClass hom G M]
    (F : ι → hom) (hf : EquicontinuousAt ((↑) ∘ F) (1 : G)) :
    UniformEquicontinuous ((↑) ∘ F) := by
  /-
    ι : Type u_1
    G : Type u_2
    M : Type u_3
    hom : Type u_4
    inst✝⁷ : UniformSpace G
    inst✝⁶ : UniformSpace M
    inst✝⁵ : Group G
    inst✝⁴ : Group M
    inst✝³ : UniformGroup G
    inst✝² : UniformGroup M
    inst✝¹ : FunLike hom G M
    inst✝ : MonoidHomClass hom G M
    F : ι → hom
    hf : EquicontinuousAt (Function.comp DFunLike.coe F) 1
    ⊢ UniformEquicontinuous (Function.comp DFunLike.coe F)
  -/
  rw [uniformEquicontinuous_iff_uniformContinuous]
  /-
    ι : Type u_1
    G : Type u_2
    M : Type u_3
    hom : Type u_4
    inst✝⁷ : UniformSpace G
    inst✝⁶ : UniformSpace M
    inst✝⁵ : Group G
    inst✝⁴ : Group M
    inst✝³ : UniformGroup G
    inst✝² : UniformGroup M
    inst✝¹ : FunLike hom G M
    inst✝ : MonoidHomClass hom G M
    F : ι → hom
    hf : EquicontinuousAt (Function.comp DFunLike.coe F) 1
    ⊢ UniformContinuous (Function.comp (⇑UniformFun.ofFun) (Function.swap (Functio …
  -/
  rw [equicontinuousAt_iff_continuousAt] at hf
  let φ : G →* (ι →ᵤ M) :=
    { toFun := swap ((↑) ∘ F)
      map_one' := by dsimp [UniformFun]; ext; exact map_one _
      map_mul' := fun a b => by dsimp [UniformFun]; ext; exact map_mul _ _ _ }
  /-
    ι : Type u_1
    G : Type u_2
    M : Type u_3
    hom : Type u_4
    inst✝⁷ : UniformSpace G
    inst✝⁶ : UniformSpace M
    inst✝⁵ : Group G
    inst✝⁴ : Group M
    inst✝³ : UniformGroup G
    inst✝² : UniformGroup M
    inst✝¹ : FunLike hom G M
    inst✝ : MonoidHomClass hom G M
    F : ι → hom
    hf : ContinuousAt (Function.comp (⇑UniformFun.ofFun) (Function.swap (Function. …
    φ : MonoidHom G (UniformFun ι M) := { toFun := Function.swap (Function.comp DF …
    ⊢ UniformContinuous (Function.comp (⇑UniformFun.ofFun) (Function.swap (Functio …
  -/
  exact uniformContinuous_of_continuousAt_one φ hf
  /-
    🎉 no goals
  -/

