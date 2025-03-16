/-- The compact-open topology on the space of continuous maps `C(X, Y)`. -/
instance compactOpen : TopologicalSpace C(X, Y) :=
  .generateFrom <| image2 (fun K U ↦ {f | MapsTo f K U}) {K | IsCompact K} {U | IsOpen U}


/-- Definition of `ContinuousMap.compactOpen`. -/
theorem compactOpen_eq : @compactOpen X Y _ _ =
    .generateFrom (image2 (fun K U ↦ {f | MapsTo f K U}) {K | IsCompact K} {t | IsOpen t}) :=
   rfl


theorem isOpen_setOf_mapsTo (hK : IsCompact K) (hU : IsOpen U) :
    IsOpen {f : C(X, Y) | MapsTo f K U} :=
  isOpen_generateFrom_of_mem <| mem_image2_of_mem hK hU


lemma eventually_mapsTo {f : C(X, Y)} (hK : IsCompact K) (hU : IsOpen U) (h : MapsTo f K U) :
    ∀ᶠ g : C(X, Y) in 𝓝 f, MapsTo g K U :=
  (isOpen_setOf_mapsTo hK hU).mem_nhds h


lemma nhds_compactOpen (f : C(X, Y)) :
    𝓝 f = ⨅ (K : Set X) (_ : IsCompact K) (U : Set Y) (_ : IsOpen U) (_ : MapsTo f K U),
      𝓟 {g : C(X, Y) | MapsTo g K U} := by
  simp_rw [compactOpen_eq, nhds_generateFrom, mem_setOf_eq, @and_comm (f ∈ _), iInf_and,
    ← image_prod, iInf_image, biInf_prod, mem_setOf_eq]


lemma tendsto_nhds_compactOpen {l : Filter α} {f : α → C(Y, Z)} {g : C(Y, Z)} :
    Tendsto f l (𝓝 g) ↔
      ∀ K, IsCompact K → ∀ U, IsOpen U → MapsTo g K U → ∀ᶠ a in l, MapsTo (f a) K U := by
  /-
    α : Type u_1
    Y : Type u_3
    Z : Type u_4
    inst✝¹ : TopologicalSpace Y
    inst✝ : TopologicalSpace Z
    l : Filter α
    f : α → ContinuousMap Y Z
    g : ContinuousMap Y Z
    ⊢ Iff (Filter.Tendsto f l (nhds g)) (∀ (K : Set Y), IsCompact K → ∀ (U : Set Z …
  -/
  simp [nhds_compactOpen]
  /-
    🎉 no goals
  -/


lemma continuous_compactOpen {f : X → C(Y, Z)} :
    Continuous f ↔ ∀ K, IsCompact K → ∀ U, IsOpen U → IsOpen {x | MapsTo (f x) K U} :=
  continuous_generateFrom_iff.trans forall_mem_image2


/-- `C(X, ·)` is a functor. -/
theorem continuous_postcomp (g : C(Y, Z)) : Continuous (ContinuousMap.comp g : C(X, Y) → C(X, Z)) :=
  continuous_compactOpen.2 fun _K hK _U hU ↦ isOpen_setOf_mapsTo hK (hU.preimage g.2)


@[deprecated (since := "2024-10-19")] alias continuous_comp := continuous_postcomp


/-- If `g : C(Y, Z)` is a topology inducing map,
then the composition `ContinuousMap.comp g : C(X, Y) → C(X, Z)` is a topology inducing map too. -/
theorem isInducing_postcomp (g : C(Y, Z)) (hg : IsInducing g) :
    IsInducing (g.comp : C(X, Y) → C(X, Z)) where
  eq_induced := by
    simp only [compactOpen_eq, induced_generateFrom_eq, image_image2, hg.setOf_isOpen,
      image2_image_right, MapsTo, mem_preimage, preimage_setOf_eq, comp_apply]


@[deprecated (since := "2024-10-28")] alias inducing_postcomp := isInducing_postcomp


@[deprecated (since := "2024-10-19")] alias inducing_comp := isInducing_postcomp


/-- If `g : C(Y, Z)` is a topological embedding,
then the composition `ContinuousMap.comp g : C(X, Y) → C(X, Z)` is an embedding too. -/
theorem isEmbedding_postcomp (g : C(Y, Z)) (hg : IsEmbedding g) :
    IsEmbedding (g.comp : C(X, Y) → C(X, Z)) :=
  ⟨isInducing_postcomp g hg.1, fun _ _ ↦ (cancel_left hg.2).1⟩


@[deprecated (since := "2024-10-26")]
alias embedding_postcomp := isEmbedding_postcomp


@[deprecated (since := "2024-10-19")] alias embedding_comp := isEmbedding_postcomp


/-- `C(·, Z)` is a functor. -/
@[continuity, fun_prop]
theorem continuous_precomp (f : C(X, Y)) : Continuous (fun g => g.comp f : C(Y, Z) → C(X, Z)) :=
  continuous_compactOpen.2 fun K hK U hU ↦ by
    /-
      X : Type u_2
      Y : Type u_3
      Z : Type u_4
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace Y
      inst✝ : TopologicalSpace Z
      f : ContinuousMap X Y
      K : Set X
      hK : IsCompact K
      U : Set Z
      hU : IsOpen U
      ⊢ IsOpen (setOf fun x => Set.MapsTo (⇑(x.comp f)) K U)
    -/
    simpa only [mapsTo_image_iff] using isOpen_setOf_mapsTo (hK.image f.2) hU
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-10-19")] alias continuous_comp_left := continuous_precomp


variable (Z) in
/-- Precomposition by a continuous map is itself a continuous map between spaces of continuous maps.
-/
@[simps apply]
def compRightContinuousMap (f : C(X, Y)) :
    C(C(Y, Z), C(X, Z)) where
  toFun g := g.comp f


/-- Any pair of homeomorphisms `X ≃ₜ Z` and `Y ≃ₜ T` gives rise to a homeomorphism
`C(X, Y) ≃ₜ C(Z, T)`. -/
protected def _root_.Homeomorph.arrowCongr (φ : X ≃ₜ Z) (ψ : Y ≃ₜ T) :
    C(X, Y) ≃ₜ C(Z, T) where
  toFun f := .comp ψ <| f.comp φ.symm
  invFun f := .comp ψ.symm <| f.comp φ
  left_inv f := ext fun _ ↦ ψ.left_inv (f _) |>.trans <| congrArg f <| φ.left_inv _
  right_inv f := ext fun _ ↦ ψ.right_inv (f _) |>.trans <| congrArg f <| φ.right_inv _
  continuous_toFun := continuous_postcomp _ |>.comp <| continuous_precomp _
  continuous_invFun := continuous_postcomp _ |>.comp <| continuous_precomp _


variable (Z) in
/-- Precomposition by a homeomorphism is itself a homeomorphism between spaces of continuous maps.
-/
@[deprecated Homeomorph.arrowCongr (since := "2024-10-19")]
def compRightHomeomorph (f : X ≃ₜ Y) :
    C(Y, Z) ≃ₜ C(X, Z) :=
  .arrowCongr f.symm (.refl _)


/-- Composition is a continuous map from `C(X, Y) × C(Y, Z)` to `C(X, Z)`,
provided that `Y` is locally compact.
This is Prop. 9 of Chap. X, §3, №. 4 of Bourbaki's *Topologie Générale*. -/
theorem continuous_comp' : Continuous fun x : C(X, Y) × C(Y, Z) => x.2.comp x.1 := by
  /-
    X : Type u_2
    Y : Type u_3
    Z : Type u_4
    inst✝³ : TopologicalSpace X
    inst✝² : TopologicalSpace Y
    inst✝¹ : TopologicalSpace Z
    inst✝ : LocallyCompactPair Y Z
    ⊢ Continuous fun x => x.2.comp x.1
  -/
  simp_rw [continuous_iff_continuousAt, ContinuousAt, tendsto_nhds_compactOpen]
  /-
    X : Type u_2
    Y : Type u_3
    Z : Type u_4
    inst✝³ : TopologicalSpace X
    inst✝² : TopologicalSpace Y
    inst✝¹ : TopologicalSpace Z
    inst✝ : LocallyCompactPair Y Z
    ⊢ ∀ (x : Prod (ContinuousMap X Y) (ContinuousMap Y Z)) (K : Set X), IsCompact  …
  -/
  intro ⟨f, g⟩ K hK U hU (hKU : MapsTo (g ∘ f) K U)
  obtain ⟨L, hKL, hLc, hLU⟩ : ∃ L ∈ 𝓝ˢ (f '' K), IsCompact L ∧ MapsTo g L U :=
    exists_mem_nhdsSet_isCompact_mapsTo g.continuous (hK.image f.continuous) hU
      (mapsTo_image_iff.2 hKU)
  /-
    case intro.intro.intro
    X : Type u_2
    Y : Type u_3
    Z : Type u_4
    inst✝³ : TopologicalSpace X
    inst✝² : TopologicalSpace Y
    inst✝¹ : TopologicalSpace Z
    inst✝ : LocallyCompactPair Y Z
    f : ContinuousMap X Y
    g : ContinuousMap Y Z
    K : Set X
    hK : IsCompact K
    U : Set Z
    hU : IsOpen U
    hKU : Set.MapsTo (Function.comp ⇑g ⇑f) K U
    L : Set Y
    hKL : Membership.mem (nhdsSet (Set.image (⇑f) K)) L
    hLc : IsCompact L
    hLU : Set.MapsTo (⇑g) L U
    ⊢ Filter.Eventually (fun a => Set.MapsTo (⇑(a.2.comp a.1)) K U) (nhds { fst := …
  -/
  rw [← subset_interior_iff_mem_nhdsSet, ← mapsTo'] at hKL
  exact ((eventually_mapsTo hK isOpen_interior hKL).prod_nhds
    (eventually_mapsTo hLc hU hLU)).mono fun ⟨f', g'⟩ ⟨hf', hg'⟩ ↦
      hg'.comp <| hf'.mono_right interior_subset


lemma _root_.Filter.Tendsto.compCM {α : Type*} {l : Filter α} {g : α → C(Y, Z)} {g₀ : C(Y, Z)}
    {f : α → C(X, Y)} {f₀ : C(X, Y)} (hg : Tendsto g l (𝓝 g₀)) (hf : Tendsto f l (𝓝 f₀)) :
    Tendsto (fun a ↦ (g a).comp (f a)) l (𝓝 (g₀.comp f₀)) :=
  (continuous_comp'.tendsto (f₀, g₀)).comp (hf.prod_mk_nhds hg)


nonrec lemma _root_.ContinuousAt.compCM (hg : ContinuousAt g a) (hf : ContinuousAt f a) :
    ContinuousAt (fun x ↦ (g x).comp (f x)) a :=
  hg.compCM hf


nonrec lemma _root_.ContinuousWithinAt.compCM (hg : ContinuousWithinAt g s a)
    (hf : ContinuousWithinAt f s a) : ContinuousWithinAt (fun x ↦ (g x).comp (f x)) s a :=
  hg.compCM hf


lemma _root_.ContinuousOn.compCM (hg : ContinuousOn g s) (hf : ContinuousOn f s) :
    ContinuousOn (fun x ↦ (g x).comp (f x)) s := fun a ha ↦
  (hg a ha).compCM (hf a ha)


lemma _root_.Continuous.compCM (hg : Continuous g) (hf : Continuous f) :
    Continuous fun x => (g x).comp (f x) :=
  continuous_comp'.comp (hf.prod_mk hg)


/-- The evaluation map `C(X, Y) × X → Y` is continuous
if `X, Y` is a locally compact pair of spaces. -/
instance [LocallyCompactPair X Y] : ContinuousEval C(X, Y) X Y where
  continuous_eval := by
    /-
      α : Type u_1
      X : Type u_2
      Y : Type u_3
      Z : Type u_4
      T : Type u_5
      inst✝⁴ : TopologicalSpace X
      inst✝³ : TopologicalSpace Y
      inst✝² : TopologicalSpace Z
      inst✝¹ : TopologicalSpace T
      K : Set X
      U : Set Y
      inst✝ : LocallyCompactPair X Y
      ⊢ Continuous fun fx => fx.1 fx.2
    -/
    simp_rw [continuous_iff_continuousAt, ContinuousAt, (nhds_basis_opens _).tendsto_right_iff]
    /-
      α : Type u_1
      X : Type u_2
      Y : Type u_3
      Z : Type u_4
      T : Type u_5
      inst✝⁴ : TopologicalSpace X
      inst✝³ : TopologicalSpace Y
      inst✝² : TopologicalSpace Z
      inst✝¹ : TopologicalSpace T
      K : Set X
      U : Set Y
      inst✝ : LocallyCompactPair X Y
      ⊢ ∀ (x : Prod (ContinuousMap X Y) X) (i : Set Y), And (Membership.mem i (x.1 x …
    -/
    rintro ⟨f, x⟩ U ⟨hx : f x ∈ U, hU : IsOpen U⟩
    /-
      case mk.intro
      α : Type u_1
      X : Type u_2
      Y : Type u_3
      Z : Type u_4
      T : Type u_5
      inst✝⁴ : TopologicalSpace X
      inst✝³ : TopologicalSpace Y
      inst✝² : TopologicalSpace Z
      inst✝¹ : TopologicalSpace T
      K : Set X
      U✝ : Set Y
      inst✝ : LocallyCompactPair X Y
      f : ContinuousMap X Y
      x : X
      U : Set Y
      hx : Membership.mem U (f x)
      hU : IsOpen U
      ⊢ Filter.Eventually (fun x => Membership.mem U (x.1 x.2)) (nhds { fst := f, sn …
    -/
    rcases exists_mem_nhds_isCompact_mapsTo f.continuous (hU.mem_nhds hx) with ⟨K, hxK, hK, hKU⟩
    /-
      case mk.intro.intro.intro.intro
      α : Type u_1
      X : Type u_2
      Y : Type u_3
      Z : Type u_4
      T : Type u_5
      inst✝⁴ : TopologicalSpace X
      inst✝³ : TopologicalSpace Y
      inst✝² : TopologicalSpace Z
      inst✝¹ : TopologicalSpace T
      K✝ : Set X
      U✝ : Set Y
      inst✝ : LocallyCompactPair X Y
      f : ContinuousMap X Y
      x : X
      U : Set Y
      hx : Membership.mem U (f x)
      hU : IsOpen U
      K : Set X
      hxK : Membership.mem (nhds x) K
      hK : IsCompact K
      hKU : Set.MapsTo (⇑f) K U
      ⊢ Filter.Eventually (fun x => Membership.mem U (x.1 x.2)) (nhds { fst := f, sn …
    -/
    filter_upwards [prod_mem_nhds (eventually_mapsTo hK hU hKU) hxK] using fun _ h ↦ h.1 h.2
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-10-01")] protected alias continuous_eval := continuous_eval


instance : ContinuousEvalConst C(X, Y) X Y where
  continuous_eval_const x :=
                                   /-
                                     α : Type u_1
                                     X : Type u_2
                                     Y : Type u_3
                                     Z : Type u_4
                                     T : Type u_5
                                     inst✝³ : TopologicalSpace X
                                     inst✝² : TopologicalSpace Y
                                     inst✝¹ : TopologicalSpace Z
                                     inst✝ : TopologicalSpace T
                                     K : Set X
                                     U✝ : Set Y
                                     x : X
                                     U : Set Y
                                     hU : IsOpen U
                                     ⊢ IsOpen (Set.preimage (fun f => f x) U)
                                   -/
    continuous_def.2 fun U hU ↦ by simpa using isOpen_setOf_mapsTo isCompact_singleton hU
                                   /-
                                     🎉 no goals
                                   -/


@[deprecated (since := "2024-10-01")] protected alias continuous_eval_const := continuous_eval_const


@[deprecated continuous_coeFun (since := "2024-10-01")]
theorem continuous_coe : Continuous ((⇑) : C(X, Y) → (X → Y)) :=
  continuous_coeFun


lemma isClosed_setOf_mapsTo {t : Set Y} (ht : IsClosed t) (s : Set X) :
    IsClosed {f : C(X, Y) | MapsTo f s t} :=
  ht.setOf_mapsTo fun _ _ ↦ continuous_eval_const _


lemma isClopen_setOf_mapsTo (hK : IsCompact K) (hU : IsClopen U) :
    IsClopen {f : C(X, Y) | MapsTo f K U} :=
  ⟨isClosed_setOf_mapsTo hU.isClosed K, isOpen_setOf_mapsTo hK hU.isOpen⟩


@[norm_cast]
lemma specializes_coe {f g : C(X, Y)} : ⇑f ⤳ ⇑g ↔ f ⤳ g := by
  /-
    X : Type u_2
    Y : Type u_3
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    f g : ContinuousMap X Y
    ⊢ Iff (Specializes ⇑f ⇑g) (Specializes f g)
  -/
  refine ⟨fun h ↦ ?_, fun h ↦ h.map continuous_coeFun⟩
  suffices ∀ K, IsCompact K → ∀ U, IsOpen U → MapsTo g K U → MapsTo f K U by
    simpa [specializes_iff_pure, nhds_compactOpen]
  /-
    X : Type u_2
    Y : Type u_3
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    f g : ContinuousMap X Y
    h : Specializes ⇑f ⇑g
    ⊢ ∀ (K : Set X), IsCompact K → ∀ (U : Set Y), IsOpen U → Set.MapsTo (⇑g) K U → …
  -/
  exact fun K _ U hU hg x hx ↦ (h.map (continuous_apply x)).mem_open hU (hg hx)
  /-
    🎉 no goals
  -/


@[norm_cast]
lemma inseparable_coe {f g : C(X, Y)} : Inseparable (f : X → Y) g ↔ Inseparable f g := by
  /-
    X : Type u_2
    Y : Type u_3
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    f g : ContinuousMap X Y
    ⊢ Iff (Inseparable ⇑f ⇑g) (Inseparable f g)
  -/
  simp only [inseparable_iff_specializes_and, specializes_coe]
  /-
    🎉 no goals
  -/


instance [T0Space Y] : T0Space C(X, Y) :=
  t0Space_of_injective_of_continuous DFunLike.coe_injective continuous_coeFun


instance [R0Space Y] : R0Space C(X, Y) where
  specializes_symmetric f g h := by
    /-
      α : Type u_1
      X : Type u_2
      Y : Type u_3
      Z : Type u_4
      T : Type u_5
      inst✝⁴ : TopologicalSpace X
      inst✝³ : TopologicalSpace Y
      inst✝² : TopologicalSpace Z
      inst✝¹ : TopologicalSpace T
      K : Set X
      U : Set Y
      inst✝ : R0Space Y
      f g : ContinuousMap X Y
      h : Specializes f g
      ⊢ Specializes g f
    -/
    rw [← specializes_coe] at h ⊢
    /-
      α : Type u_1
      X : Type u_2
      Y : Type u_3
      Z : Type u_4
      T : Type u_5
      inst✝⁴ : TopologicalSpace X
      inst✝³ : TopologicalSpace Y
      inst✝² : TopologicalSpace Z
      inst✝¹ : TopologicalSpace T
      K : Set X
      U : Set Y
      inst✝ : R0Space Y
      f g : ContinuousMap X Y
      h : Specializes ⇑f ⇑g
      ⊢ Specializes ⇑g ⇑f
    -/
    exact h.symm
    /-
      🎉 no goals
    -/


instance [T1Space Y] : T1Space C(X, Y) :=
  t1Space_of_injective_of_continuous DFunLike.coe_injective continuous_coeFun


instance [R1Space Y] : R1Space C(X, Y) :=
  .of_continuous_specializes_imp continuous_coeFun fun _ _ ↦ specializes_coe.1


instance [T2Space Y] : T2Space C(X, Y) := inferInstance


instance [RegularSpace Y] : RegularSpace C(X, Y) :=
  .of_lift'_closure_le fun f ↦ by
    /-
      α : Type u_1
      X : Type u_2
      Y : Type u_3
      Z : Type u_4
      T : Type u_5
      inst✝⁴ : TopologicalSpace X
      inst✝³ : TopologicalSpace Y
      inst✝² : TopologicalSpace Z
      inst✝¹ : TopologicalSpace T
      K : Set X
      U : Set Y
      inst✝ : RegularSpace Y
      f : ContinuousMap X Y
      ⊢ LE.le ((nhds f).lift' closure) (nhds f)
    -/
    rw [← tendsto_id', tendsto_nhds_compactOpen]
    /-
      α : Type u_1
      X : Type u_2
      Y : Type u_3
      Z : Type u_4
      T : Type u_5
      inst✝⁴ : TopologicalSpace X
      inst✝³ : TopologicalSpace Y
      inst✝² : TopologicalSpace Z
      inst✝¹ : TopologicalSpace T
      K : Set X
      U : Set Y
      inst✝ : RegularSpace Y
      f : ContinuousMap X Y
      ⊢ ∀ (K : Set X), IsCompact K → ∀ (U : Set Y), IsOpen U → Set.MapsTo (⇑f) K U → …
    -/
    intro K hK U hU hf
    rcases (hK.image f.continuous).exists_isOpen_closure_subset (hU.mem_nhdsSet.2 hf.image_subset)
      with ⟨V, hVo, hKV, hVU⟩
    /-
      case intro.intro.intro
      α : Type u_1
      X : Type u_2
      Y : Type u_3
      Z : Type u_4
      T : Type u_5
      inst✝⁴ : TopologicalSpace X
      inst✝³ : TopologicalSpace Y
      inst✝² : TopologicalSpace Z
      inst✝¹ : TopologicalSpace T
      K✝ : Set X
      U✝ : Set Y
      inst✝ : RegularSpace Y
      f : ContinuousMap X Y
      K : Set X
      hK : IsCompact K
      U : Set Y
      hU : IsOpen U
      hf : Set.MapsTo (⇑f) K U
      V : Set Y
      hVo : IsOpen V
      hKV : HasSubset.Subset (Set.image (⇑f) K) V
      hVU : HasSubset.Subset (closure V) U
      ⊢ Filter.Eventually (fun a => Set.MapsTo (⇑(id a)) K U) ((nhds f).lift' closure)
    -/
    filter_upwards [mem_lift' (eventually_mapsTo hK hVo (mapsTo'.2 hKV))] with g hg
    /-
      case h
      α : Type u_1
      X : Type u_2
      Y : Type u_3
      Z : Type u_4
      T : Type u_5
      inst✝⁴ : TopologicalSpace X
      inst✝³ : TopologicalSpace Y
      inst✝² : TopologicalSpace Z
      inst✝¹ : TopologicalSpace T
      K✝ : Set X
      U✝ : Set Y
      inst✝ : RegularSpace Y
      f : ContinuousMap X Y
      K : Set X
      hK : IsCompact K
      U : Set Y
      hU : IsOpen U
      hf : Set.MapsTo (⇑f) K U
      V : Set Y
      hVo : IsOpen V
      hKV : HasSubset.Subset (Set.image (⇑f) K) V
      hVU : HasSubset.Subset (closure V) U
      g : ContinuousMap X Y
      hg : Membership.mem (closure (setOf fun x => Set.MapsTo (⇑x) K V)) g
      ⊢ Set.MapsTo (⇑(id g)) K U
    -/
    refine ((isClosed_setOf_mapsTo isClosed_closure K).closure_subset ?_).mono_right hVU
    /-
      case h
      α : Type u_1
      X : Type u_2
      Y : Type u_3
      Z : Type u_4
      T : Type u_5
      inst✝⁴ : TopologicalSpace X
      inst✝³ : TopologicalSpace Y
      inst✝² : TopologicalSpace Z
      inst✝¹ : TopologicalSpace T
      K✝ : Set X
      U✝ : Set Y
      inst✝ : RegularSpace Y
      f : ContinuousMap X Y
      K : Set X
      hK : IsCompact K
      U : Set Y
      hU : IsOpen U
      hf : Set.MapsTo (⇑f) K U
      V : Set Y
      hVo : IsOpen V
      hKV : HasSubset.Subset (Set.image (⇑f) K) V
      hVU : HasSubset.Subset (closure V) U
      g : ContinuousMap X Y
      hg : Membership.mem (closure (setOf fun x => Set.MapsTo (⇑x) K V)) g
      ⊢ Membership.mem (closure (setOf fun f => Set.MapsTo (⇑f) K (closure V))) (id g)
    -/
    exact closure_mono (fun _ h ↦ h.mono_right subset_closure) hg
    /-
      🎉 no goals
    -/


instance [T3Space Y] : T3Space C(X, Y) := inferInstance


/-- For any subset `s` of `X`, the restriction of continuous functions to `s` is continuous
as a function from `C(X, Y)` to `C(s, Y)` with their respective compact-open topologies. -/
theorem continuous_restrict (s : Set X) : Continuous fun F : C(X, Y) => F.restrict s :=
  continuous_precomp <| restrict s <| .id X


theorem compactOpen_le_induced (s : Set X) :
    (ContinuousMap.compactOpen : TopologicalSpace C(X, Y)) ≤
      .induced (restrict s) ContinuousMap.compactOpen :=
  (continuous_restrict s).le_induced


/-- The compact-open topology on `C(X, Y)`
is equal to the infimum of the compact-open topologies on `C(s, Y)` for `s` a compact subset of `X`.
The key point of the proof is that for every compact set `K`,
the universal set `Set.univ : Set K` is a compact set as well. -/
theorem compactOpen_eq_iInf_induced :
    (ContinuousMap.compactOpen : TopologicalSpace C(X, Y)) =
      ⨅ (K : Set X) (_ : IsCompact K), .induced (.restrict K) ContinuousMap.compactOpen := by
  /-
    X : Type u_2
    Y : Type u_3
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    ⊢ Eq ContinuousMap.compactOpen (iInf fun K => iInf fun x => TopologicalSpace.i …
  -/
  refine le_antisymm (le_iInf₂ fun s _ ↦ compactOpen_le_induced s) ?_
  /-
    X : Type u_2
    Y : Type u_3
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    ⊢ LE.le (iInf fun K => iInf fun x => TopologicalSpace.induced (ContinuousMap.r …
  -/
  refine le_generateFrom <| forall_mem_image2.2 fun K (hK : IsCompact K) U hU ↦ ?_
  /-
    X : Type u_2
    Y : Type u_3
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    K : Set X
    hK : IsCompact K
    U : Set Y
    hU : Membership.mem (setOf fun U => IsOpen U) U
    ⊢ IsOpen (setOf fun f => Set.MapsTo (⇑f) K U)
  -/
  refine TopologicalSpace.le_def.1 (iInf₂_le K hK) _ ?_
  /-
    X : Type u_2
    Y : Type u_3
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    K : Set X
    hK : IsCompact K
    U : Set Y
    hU : Membership.mem (setOf fun U => IsOpen U) U
    ⊢ IsOpen (setOf fun f => Set.MapsTo (⇑f) K U)
  -/
  convert isOpen_induced (isOpen_setOf_mapsTo (isCompact_iff_isCompact_univ.1 hK) hU)
  /-
    case h.e'_3
    X : Type u_2
    Y : Type u_3
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    K : Set X
    hK : IsCompact K
    U : Set Y
    hU : Membership.mem (setOf fun U => IsOpen U) U
    ⊢ Eq (setOf fun f => Set.MapsTo (⇑f) K U) (Set.preimage (ContinuousMap.restric …
  -/
  simp [mapsTo_univ_iff, Subtype.forall, MapsTo]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-03-05")]
alias compactOpen_eq_sInf_induced := compactOpen_eq_iInf_induced


theorem nhds_compactOpen_eq_iInf_nhds_induced (f : C(X, Y)) :
    𝓝 f = ⨅ (s) (_ : IsCompact s), (𝓝 (f.restrict s)).comap (ContinuousMap.restrict s) := by
  /-
    X : Type u_2
    Y : Type u_3
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    f : ContinuousMap X Y
    ⊢ Eq (nhds f) (iInf fun s => iInf fun x => Filter.comap (ContinuousMap.restric …
  -/
  rw [compactOpen_eq_iInf_induced]
  /-
    X : Type u_2
    Y : Type u_3
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    f : ContinuousMap X Y
    ⊢ Eq (nhds f) (iInf fun s => iInf fun x => Filter.comap (ContinuousMap.restric …
  -/
  simp only [nhds_iInf, nhds_induced]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-03-05")]
alias nhds_compactOpen_eq_sInf_nhds_induced := nhds_compactOpen_eq_iInf_nhds_induced


theorem tendsto_compactOpen_restrict {ι : Type*} {l : Filter ι} {F : ι → C(X, Y)} {f : C(X, Y)}
    (hFf : Filter.Tendsto F l (𝓝 f)) (s : Set X) :
    Tendsto (fun i => (F i).restrict s) l (𝓝 (f.restrict s)) :=
  (continuous_restrict s).continuousAt.tendsto.comp hFf


theorem tendsto_compactOpen_iff_forall {ι : Type*} {l : Filter ι} (F : ι → C(X, Y)) (f : C(X, Y)) :
    Tendsto F l (𝓝 f) ↔
      ∀ K, IsCompact K → Tendsto (fun i => (F i).restrict K) l (𝓝 (f.restrict K)) := by
  /-
    X : Type u_2
    Y : Type u_3
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    ι : Type u_6
    l : Filter ι
    F : ι → ContinuousMap X Y
    f : ContinuousMap X Y
    ⊢ Iff (Filter.Tendsto F l (nhds f)) (∀ (K : Set X), IsCompact K → Filter.Tends …
  -/
  rw [compactOpen_eq_iInf_induced]
  /-
    X : Type u_2
    Y : Type u_3
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    ι : Type u_6
    l : Filter ι
    F : ι → ContinuousMap X Y
    f : ContinuousMap X Y
    ⊢ Iff (Filter.Tendsto F l (nhds f)) (∀ (K : Set X), IsCompact K → Filter.Tends …
  -/
  simp [nhds_iInf, nhds_induced, Filter.tendsto_comap_iff, Function.comp_def]
  /-
    🎉 no goals
  -/


/-- A family `F` of functions in `C(X, Y)` converges in the compact-open topology, if and only if
it converges in the compact-open topology on each compact subset of `X`. -/
theorem exists_tendsto_compactOpen_iff_forall [WeaklyLocallyCompactSpace X] [T2Space Y]
    {ι : Type*} {l : Filter ι} [Filter.NeBot l] (F : ι → C(X, Y)) :
    (∃ f, Filter.Tendsto F l (𝓝 f)) ↔
      ∀ s : Set X, IsCompact s → ∃ f, Filter.Tendsto (fun i => (F i).restrict s) l (𝓝 f) := by
  /-
    X : Type u_2
    Y : Type u_3
    inst✝⁴ : TopologicalSpace X
    inst✝³ : TopologicalSpace Y
    inst✝² : WeaklyLocallyCompactSpace X
    inst✝¹ : T2Space Y
    ι : Type u_6
    l : Filter ι
    inst✝ : l.NeBot
    F : ι → ContinuousMap X Y
    ⊢ Iff (Exists fun f => Filter.Tendsto F l (nhds f)) (∀ (s : Set X), IsCompact  …
  -/
  constructor
    /-
      case mp
      X : Type u_2
      Y : Type u_3
      inst✝⁴ : TopologicalSpace X
      inst✝³ : TopologicalSpace Y
      inst✝² : WeaklyLocallyCompactSpace X
      inst✝¹ : T2Space Y
      ι : Type u_6
      l : Filter ι
      inst✝ : l.NeBot
      F : ι → ContinuousMap X Y
      ⊢ (Exists fun f => Filter.Tendsto F l (nhds f)) → ∀ (s : Set X), IsCompact s → …
    -/
  · rintro ⟨f, hf⟩ s _
    /-
      case mp.intro
      X : Type u_2
      Y : Type u_3
      inst✝⁴ : TopologicalSpace X
      inst✝³ : TopologicalSpace Y
      inst✝² : WeaklyLocallyCompactSpace X
      inst✝¹ : T2Space Y
      ι : Type u_6
      l : Filter ι
      inst✝ : l.NeBot
      F : ι → ContinuousMap X Y
      f : ContinuousMap X Y
      hf : Filter.Tendsto F l (nhds f)
      s : Set X
      a✝ : IsCompact s
      ⊢ Exists fun f => Filter.Tendsto (fun i => ContinuousMap.restrict s (F i)) l ( …
    -/
    exact ⟨f.restrict s, tendsto_compactOpen_restrict hf s⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      X : Type u_2
      Y : Type u_3
      inst✝⁴ : TopologicalSpace X
      inst✝³ : TopologicalSpace Y
      inst✝² : WeaklyLocallyCompactSpace X
      inst✝¹ : T2Space Y
      ι : Type u_6
      l : Filter ι
      inst✝ : l.NeBot
      F : ι → ContinuousMap X Y
      ⊢ (∀ (s : Set X), IsCompact s → Exists fun f => Filter.Tendsto (fun i => Conti …
    -/
  · intro h
    /-
      case mpr
      X : Type u_2
      Y : Type u_3
      inst✝⁴ : TopologicalSpace X
      inst✝³ : TopologicalSpace Y
      inst✝² : WeaklyLocallyCompactSpace X
      inst✝¹ : T2Space Y
      ι : Type u_6
      l : Filter ι
      inst✝ : l.NeBot
      F : ι → ContinuousMap X Y
      h : ∀ (s : Set X), IsCompact s → Exists fun f => Filter.Tendsto (fun i => Cont …
      ⊢ Exists fun f => Filter.Tendsto F l (nhds f)
    -/
    choose f hf using h
    -- By uniqueness of limits in a `T2Space`, since `fun i ↦ F i x` tends to both `f s₁ hs₁ x` and
    -- `f s₂ hs₂ x`, we have `f s₁ hs₁ x = f s₂ hs₂ x`
    have h :
      ∀ (s₁) (hs₁ : IsCompact s₁) (s₂) (hs₂ : IsCompact s₂) (x : X) (hxs₁ : x ∈ s₁) (hxs₂ : x ∈ s₂),
        f s₁ hs₁ ⟨x, hxs₁⟩ = f s₂ hs₂ ⟨x, hxs₂⟩ := by
      rintro s₁ hs₁ s₂ hs₂ x hxs₁ hxs₂
      haveI := isCompact_iff_compactSpace.mp hs₁
      haveI := isCompact_iff_compactSpace.mp hs₂
      have h₁ := (continuous_eval_const (⟨x, hxs₁⟩ : s₁)).continuousAt.tendsto.comp (hf s₁ hs₁)
      have h₂ := (continuous_eval_const (⟨x, hxs₂⟩ : s₂)).continuousAt.tendsto.comp (hf s₂ hs₂)
      exact tendsto_nhds_unique h₁ h₂
    -- So glue the `f s hs` together and prove that this glued function `f₀` is a limit on each
    -- compact set `s`
    /-
      case mpr
      X : Type u_2
      Y : Type u_3
      inst✝⁴ : TopologicalSpace X
      inst✝³ : TopologicalSpace Y
      inst✝² : WeaklyLocallyCompactSpace X
      inst✝¹ : T2Space Y
      ι : Type u_6
      l : Filter ι
      inst✝ : l.NeBot
      F : ι → ContinuousMap X Y
      f : (s : Set X) → IsCompact s → ContinuousMap (↑s) Y
      hf : ∀ (s : Set X) (a : IsCompact s), Filter.Tendsto (fun i => ContinuousMap.r …
      h : ∀ (s₁ : Set X) (hs₁ : IsCompact s₁) (s₂ : Set X) (hs₂ : IsCompact s₂) (x : …
      ⊢ Exists fun f => Filter.Tendsto F l (nhds f)
    -/
    refine ⟨liftCover' _ _ h exists_compact_mem_nhds, ?_⟩
    /-
      case mpr
      X : Type u_2
      Y : Type u_3
      inst✝⁴ : TopologicalSpace X
      inst✝³ : TopologicalSpace Y
      inst✝² : WeaklyLocallyCompactSpace X
      inst✝¹ : T2Space Y
      ι : Type u_6
      l : Filter ι
      inst✝ : l.NeBot
      F : ι → ContinuousMap X Y
      f : (s : Set X) → IsCompact s → ContinuousMap (↑s) Y
      hf : ∀ (s : Set X) (a : IsCompact s), Filter.Tendsto (fun i => ContinuousMap.r …
      h : ∀ (s₁ : Set X) (hs₁ : IsCompact s₁) (s₂ : Set X) (hs₂ : IsCompact s₂) (x : …
      ⊢ Filter.Tendsto F l (nhds (ContinuousMap.liftCover' (fun s => ∀ ⦃f : Filter X …
    -/
    rw [tendsto_compactOpen_iff_forall]
    /-
      case mpr
      X : Type u_2
      Y : Type u_3
      inst✝⁴ : TopologicalSpace X
      inst✝³ : TopologicalSpace Y
      inst✝² : WeaklyLocallyCompactSpace X
      inst✝¹ : T2Space Y
      ι : Type u_6
      l : Filter ι
      inst✝ : l.NeBot
      F : ι → ContinuousMap X Y
      f : (s : Set X) → IsCompact s → ContinuousMap (↑s) Y
      hf : ∀ (s : Set X) (a : IsCompact s), Filter.Tendsto (fun i => ContinuousMap.r …
      h : ∀ (s₁ : Set X) (hs₁ : IsCompact s₁) (s₂ : Set X) (hs₂ : IsCompact s₂) (x : …
      ⊢ ∀ (K : Set X), IsCompact K → Filter.Tendsto (fun i => ContinuousMap.restrict …
    -/
    intro s hs
    /-
      case mpr
      X : Type u_2
      Y : Type u_3
      inst✝⁴ : TopologicalSpace X
      inst✝³ : TopologicalSpace Y
      inst✝² : WeaklyLocallyCompactSpace X
      inst✝¹ : T2Space Y
      ι : Type u_6
      l : Filter ι
      inst✝ : l.NeBot
      F : ι → ContinuousMap X Y
      f : (s : Set X) → IsCompact s → ContinuousMap (↑s) Y
      hf : ∀ (s : Set X) (a : IsCompact s), Filter.Tendsto (fun i => ContinuousMap.r …
      h : ∀ (s₁ : Set X) (hs₁ : IsCompact s₁) (s₂ : Set X) (hs₂ : IsCompact s₂) (x : …
      s : Set X
      hs : IsCompact s
      ⊢ Filter.Tendsto (fun i => ContinuousMap.restrict s (F i)) l (nhds (Continuous …
    -/
    rw [liftCover_restrict']
    /-
      case mpr
      X : Type u_2
      Y : Type u_3
      inst✝⁴ : TopologicalSpace X
      inst✝³ : TopologicalSpace Y
      inst✝² : WeaklyLocallyCompactSpace X
      inst✝¹ : T2Space Y
      ι : Type u_6
      l : Filter ι
      inst✝ : l.NeBot
      F : ι → ContinuousMap X Y
      f : (s : Set X) → IsCompact s → ContinuousMap (↑s) Y
      hf : ∀ (s : Set X) (a : IsCompact s), Filter.Tendsto (fun i => ContinuousMap.r …
      h : ∀ (s₁ : Set X) (hs₁ : IsCompact s₁) (s₂ : Set X) (hs₂ : IsCompact s₂) (x : …
      s : Set X
      hs : IsCompact s
      ⊢ Filter.Tendsto (fun i => ContinuousMap.restrict s (F i)) l (nhds (f s ?m.633 …
    -/
    exact hf s hs
    /-
      🎉 no goals
    -/


/-- The coevaluation map `Y → C(X, Y × X)` sending a point `x : Y` to the continuous function
on `X` sending `y` to `(x, y)`. -/
@[simps (config := .asFn)]
def coev (b : Y) : C(X, Y × X) :=
  { toFun := Prod.mk b }


                                                                          /-
                                                                            X : Type u_2
                                                                            Y : Type u_3
                                                                            inst✝¹ : TopologicalSpace X
                                                                            inst✝ : TopologicalSpace Y
                                                                            y : Y
                                                                            s : Set X
                                                                            ⊢ Eq (Set.image (⇑(ContinuousMap.coev X Y y)) s) (SProd.sprod (Singleton.singl …
                                                                          -/
theorem image_coev {y : Y} (s : Set X) : coev X Y y '' s = {y} ×ˢ s := by simp
                                                                          /-
                                                                            🎉 no goals
                                                                          -/


/-- The coevaluation map `Y → C(X, Y × X)` is continuous (always). -/
theorem continuous_coev : Continuous (coev X Y) := by
  /-
    X : Type u_2
    Y : Type u_3
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    ⊢ Continuous (ContinuousMap.coev X Y)
  -/
  have : ∀ {a K U}, MapsTo (coev X Y a) K U ↔ {a} ×ˢ K ⊆ U := by simp [mapsTo']
  /-
    X : Type u_2
    Y : Type u_3
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    this : ∀ {a : Y} {K : Set X} {U : Set (Prod Y X)}, Iff (Set.MapsTo (⇑(Continuo …
    ⊢ Continuous (ContinuousMap.coev X Y)
  -/
  simp only [continuous_iff_continuousAt, ContinuousAt, tendsto_nhds_compactOpen, this]
  /-
    X : Type u_2
    Y : Type u_3
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    this : ∀ {a : Y} {K : Set X} {U : Set (Prod Y X)}, Iff (Set.MapsTo (⇑(Continuo …
    ⊢ ∀ (x : Y) (K : Set X), IsCompact K → ∀ (U : Set (Prod Y X)), IsOpen U → HasS …
  -/
  intro x K hK U hU hKU
  /-
    X : Type u_2
    Y : Type u_3
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    this : ∀ {a : Y} {K : Set X} {U : Set (Prod Y X)}, Iff (Set.MapsTo (⇑(Continuo …
    x : Y
    K : Set X
    hK : IsCompact K
    U : Set (Prod Y X)
    hU : IsOpen U
    hKU : HasSubset.Subset (SProd.sprod (Singleton.singleton x) K) U
    ⊢ Filter.Eventually (fun a => HasSubset.Subset (SProd.sprod (Singleton.singlet …
  -/
  rcases generalized_tube_lemma isCompact_singleton hK hU hKU with ⟨V, W, hV, -, hxV, hKW, hVWU⟩
  /-
    case intro.intro.intro.intro.intro.intro
    X : Type u_2
    Y : Type u_3
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    this : ∀ {a : Y} {K : Set X} {U : Set (Prod Y X)}, Iff (Set.MapsTo (⇑(Continuo …
    x : Y
    K : Set X
    hK : IsCompact K
    U : Set (Prod Y X)
    hU : IsOpen U
    hKU : HasSubset.Subset (SProd.sprod (Singleton.singleton x) K) U
    V : Set Y
    W : Set X
    hV : IsOpen V
    hxV : HasSubset.Subset (Singleton.singleton x) V
    hKW : HasSubset.Subset K W
    hVWU : HasSubset.Subset (SProd.sprod V W) U
    ⊢ Filter.Eventually (fun a => HasSubset.Subset (SProd.sprod (Singleton.singlet …
  -/
  filter_upwards [hV.mem_nhds (hxV rfl)] with a ha
  /-
    case h
    X : Type u_2
    Y : Type u_3
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    this : ∀ {a : Y} {K : Set X} {U : Set (Prod Y X)}, Iff (Set.MapsTo (⇑(Continuo …
    x : Y
    K : Set X
    hK : IsCompact K
    U : Set (Prod Y X)
    hU : IsOpen U
    hKU : HasSubset.Subset (SProd.sprod (Singleton.singleton x) K) U
    V : Set Y
    W : Set X
    hV : IsOpen V
    hxV : HasSubset.Subset (Singleton.singleton x) V
    hKW : HasSubset.Subset K W
    hVWU : HasSubset.Subset (SProd.sprod V W) U
    a : Y
    ha : Membership.mem V a
    ⊢ HasSubset.Subset (SProd.sprod (Singleton.singleton a) K) U
  -/
  exact (prod_mono (singleton_subset_iff.mpr ha) hKW).trans hVWU
  /-
    🎉 no goals
  -/


/-- The curried form of a continuous map `α × β → γ` as a continuous map `α → C(β, γ)`.
    If `a × β` is locally compact, this is continuous. If `α` and `β` are both locally
    compact, then this is a homeomorphism, see `Homeomorph.curry`. -/
def curry (f : C(X × Y, Z)) : C(X, C(Y, Z)) where
                                                          /-
                                                            α : Type u_1
                                                            X : Type u_2
                                                            Y : Type u_3
                                                            Z : Type u_4
                                                            T : Type u_5
                                                            inst✝³ : TopologicalSpace X
                                                            inst✝² : TopologicalSpace Y
                                                            inst✝¹ : TopologicalSpace Z
                                                            inst✝ : TopologicalSpace T
                                                            K : Set X
                                                            U : Set Y
                                                            f : ContinuousMap (Prod X Y) Z
                                                            a : X
                                                            ⊢ Continuous (Prod.mk a)
                                                          -/
  toFun a := ⟨Function.curry f a, f.continuous.comp <| by fun_prop⟩
                                                          /-
                                                            🎉 no goals
                                                          -/
  continuous_toFun := (continuous_postcomp f).comp continuous_coev


@[simp]
theorem curry_apply (f : C(X × Y, Z)) (a : X) (b : Y) : f.curry a b = f (a, b) :=
  rfl


/-- Auxiliary definition, see `ContinuousMap.curry` and `Homeomorph.curry`. -/
@[deprecated ContinuousMap.curry (since := "2024-03-05")]
def curry' (f : C(X × Y, Z)) (a : X) : C(Y, Z) := curry f a


set_option linter.deprecated false in
/-- If a map `α × β → γ` is continuous, then its curried form `α → C(β, γ)` is continuous. -/
@[deprecated ContinuousMap.curry (since := "2024-03-05")]
theorem continuous_curry' (f : C(X × Y, Z)) : Continuous (curry' f) := (curry f).continuous


/-- To show continuity of a map `α → C(β, γ)`, it suffices to show that its uncurried form
    `α × β → γ` is continuous. -/
theorem continuous_of_continuous_uncurry (f : X → C(Y, Z))
    (h : Continuous (Function.uncurry fun x y => f x y)) : Continuous f :=
  (curry ⟨_, h⟩).2


/-- The currying process is a continuous map between function spaces. -/
theorem continuous_curry [LocallyCompactSpace (X × Y)] :
    Continuous (curry : C(X × Y, Z) → C(X, C(Y, Z))) := by
  /-
    X : Type u_2
    Y : Type u_3
    Z : Type u_4
    inst✝³ : TopologicalSpace X
    inst✝² : TopologicalSpace Y
    inst✝¹ : TopologicalSpace Z
    inst✝ : LocallyCompactSpace (Prod X Y)
    ⊢ Continuous ContinuousMap.curry
  -/
  apply continuous_of_continuous_uncurry
  /-
    case h
    X : Type u_2
    Y : Type u_3
    Z : Type u_4
    inst✝³ : TopologicalSpace X
    inst✝² : TopologicalSpace Y
    inst✝¹ : TopologicalSpace Z
    inst✝ : LocallyCompactSpace (Prod X Y)
    ⊢ Continuous (Function.uncurry fun x y => x.curry y)
  -/
  apply continuous_of_continuous_uncurry
  /-
    case h.h
    X : Type u_2
    Y : Type u_3
    Z : Type u_4
    inst✝³ : TopologicalSpace X
    inst✝² : TopologicalSpace Y
    inst✝¹ : TopologicalSpace Z
    inst✝ : LocallyCompactSpace (Prod X Y)
    ⊢ Continuous (Function.uncurry fun x y => (Function.uncurry (fun x y => x.curr …
  -/
  rw [← (Homeomorph.prodAssoc _ _ _).symm.comp_continuous_iff']
  /-
    case h.h
    X : Type u_2
    Y : Type u_3
    Z : Type u_4
    inst✝³ : TopologicalSpace X
    inst✝² : TopologicalSpace Y
    inst✝¹ : TopologicalSpace Z
    inst✝ : LocallyCompactSpace (Prod X Y)
    ⊢ Continuous (Function.comp (Function.uncurry fun x y => (Function.uncurry (fu …
  -/
  exact continuous_eval
  /-
    🎉 no goals
  -/


/-- The uncurried form of a continuous map `X → C(Y, Z)` is a continuous map `X × Y → Z`. -/
theorem continuous_uncurry_of_continuous [LocallyCompactSpace Y] (f : C(X, C(Y, Z))) :
    Continuous (Function.uncurry fun x y => f x y) :=
  continuous_eval.comp <| f.continuous.prodMap continuous_id


/-- The uncurried form of a continuous map `X → C(Y, Z)` as a continuous map `X × Y → Z` (if `Y` is
    locally compact). If `X` is also locally compact, then this is a homeomorphism between the two
    function spaces, see `Homeomorph.curry`. -/
@[simps]
def uncurry [LocallyCompactSpace Y] (f : C(X, C(Y, Z))) : C(X × Y, Z) :=
  ⟨_, continuous_uncurry_of_continuous f⟩


/-- The uncurrying process is a continuous map between function spaces. -/
theorem continuous_uncurry [LocallyCompactSpace X] [LocallyCompactSpace Y] :
    Continuous (uncurry : C(X, C(Y, Z)) → C(X × Y, Z)) := by
  /-
    X : Type u_2
    Y : Type u_3
    Z : Type u_4
    inst✝⁴ : TopologicalSpace X
    inst✝³ : TopologicalSpace Y
    inst✝² : TopologicalSpace Z
    inst✝¹ : LocallyCompactSpace X
    inst✝ : LocallyCompactSpace Y
    ⊢ Continuous ContinuousMap.uncurry
  -/
  apply continuous_of_continuous_uncurry
  /-
    case h
    X : Type u_2
    Y : Type u_3
    Z : Type u_4
    inst✝⁴ : TopologicalSpace X
    inst✝³ : TopologicalSpace Y
    inst✝² : TopologicalSpace Z
    inst✝¹ : LocallyCompactSpace X
    inst✝ : LocallyCompactSpace Y
    ⊢ Continuous (Function.uncurry fun x y => x.uncurry y)
  -/
  rw [← (Homeomorph.prodAssoc _ _ _).comp_continuous_iff']
  /-
    case h
    X : Type u_2
    Y : Type u_3
    Z : Type u_4
    inst✝⁴ : TopologicalSpace X
    inst✝³ : TopologicalSpace Y
    inst✝² : TopologicalSpace Z
    inst✝¹ : LocallyCompactSpace X
    inst✝ : LocallyCompactSpace Y
    ⊢ Continuous (Function.comp (Function.uncurry fun x y => x.uncurry y) ⇑(Homeom …
  -/
  dsimp [Function.comp_def]
  /-
    case h
    X : Type u_2
    Y : Type u_3
    Z : Type u_4
    inst✝⁴ : TopologicalSpace X
    inst✝³ : TopologicalSpace Y
    inst✝² : TopologicalSpace Z
    inst✝¹ : LocallyCompactSpace X
    inst✝ : LocallyCompactSpace Y
    ⊢ Continuous fun x => Function.uncurry (fun x y => Function.uncurry (fun x_1 y …
  -/
  exact (continuous_fst.fst.eval continuous_fst.snd).eval continuous_snd
  /-
    🎉 no goals
  -/


/-- The family of constant maps: `Y → C(X, Y)` as a continuous map. -/
def const' : C(Y, C(X, Y)) :=
  curry ContinuousMap.fst


@[simp]
theorem coe_const' : (const' : Y → C(X, Y)) = const X :=
  rfl


theorem continuous_const' : Continuous (const X : Y → C(X, Y)) :=
  const'.continuous


/-- Currying as a homeomorphism between the function spaces `C(X × Y, Z)` and `C(X, C(Y, Z))`. -/
def curry [LocallyCompactSpace X] [LocallyCompactSpace Y] : C(X × Y, Z) ≃ₜ C(X, C(Y, Z)) :=
                                     /-
                                       X : Type u_1
                                       Y : Type u_2
                                       Z : Type u_3
                                       inst✝⁴ : TopologicalSpace X
                                       inst✝³ : TopologicalSpace Y
                                       inst✝² : TopologicalSpace Z
                                       inst✝¹ : LocallyCompactSpace X
                                       inst✝ : LocallyCompactSpace Y
                                       ⊢ Function.LeftInverse ContinuousMap.uncurry ContinuousMap.curry
                                     -/
                                                 /-
                                                   🎉 no goals
                                                 -/
  ⟨⟨ContinuousMap.curry, uncurry, by intro; ext; rfl, by intro; ext; rfl⟩,
                                                                     /-
                                                                       🎉 no goals
                                                                     -/
    continuous_curry, continuous_uncurry⟩


/-- If `X` has a single element, then `Y` is homeomorphic to `C(X, Y)`. -/
def continuousMapOfUnique [Unique X] : Y ≃ₜ C(X, Y) where
  toFun := const X
  invFun f := f default
  left_inv _ := rfl
  right_inv f := by
    /-
      X : Type u_1
      Y : Type u_2
      Z : Type u_3
      inst✝³ : TopologicalSpace X
      inst✝² : TopologicalSpace Y
      inst✝¹ : TopologicalSpace Z
      inst✝ : Unique X
      f : ContinuousMap X Y
      ⊢ Eq (ContinuousMap.const X ((fun f => f Inhabited.default) f)) f
    -/
    ext x
    /-
      case h
      X : Type u_1
      Y : Type u_2
      Z : Type u_3
      inst✝³ : TopologicalSpace X
      inst✝² : TopologicalSpace Y
      inst✝¹ : TopologicalSpace Z
      inst✝ : Unique X
      f : ContinuousMap X Y
      x : X
      ⊢ Eq ((ContinuousMap.const X ((fun f => f Inhabited.default) f)) x) (f x)
    -/
    rw [Unique.eq_default x]
    /-
      case h
      X : Type u_1
      Y : Type u_2
      Z : Type u_3
      inst✝³ : TopologicalSpace X
      inst✝² : TopologicalSpace Y
      inst✝¹ : TopologicalSpace Z
      inst✝ : Unique X
      f : ContinuousMap X Y
      x : X
      ⊢ Eq ((ContinuousMap.const X ((fun f => f Inhabited.default) f)) Inhabited.def …
    -/
    rfl
    /-
      🎉 no goals
    -/
  continuous_toFun := continuous_const'
  continuous_invFun := continuous_eval_const _


@[simp]
theorem continuousMapOfUnique_apply [Unique X] (y : Y) (x : X) : continuousMapOfUnique y x = y :=
  rfl


@[simp]
theorem continuousMapOfUnique_symm_apply [Unique X] (f : C(X, Y)) :
    continuousMapOfUnique.symm f = f default :=
  rfl


theorem Topology.IsQuotientMap.continuous_lift_prod_left (hf : IsQuotientMap f) {g : X × Y → Z}
    (hg : Continuous fun p : X₀ × Y => g (f p.1, p.2)) : Continuous g := by
  /-
    X₀ : Type u_1
    X : Type u_2
    Y : Type u_3
    Z : Type u_4
    inst✝⁴ : TopologicalSpace X₀
    inst✝³ : TopologicalSpace X
    inst✝² : TopologicalSpace Y
    inst✝¹ : TopologicalSpace Z
    inst✝ : LocallyCompactSpace Y
    f : X₀ → X
    hf : Topology.IsQuotientMap f
    g : Prod X Y → Z
    hg : Continuous fun p => g { fst := f p.1, snd := p.2 }
    ⊢ Continuous g
  -/
  let Gf : C(X₀, C(Y, Z)) := ContinuousMap.curry ⟨_, hg⟩
  have h : ∀ x : X, Continuous fun y => g (x, y) := by
    intro x
    obtain ⟨x₀, rfl⟩ := hf.surjective x
    exact (Gf x₀).continuous
  /-
    X₀ : Type u_1
    X : Type u_2
    Y : Type u_3
    Z : Type u_4
    inst✝⁴ : TopologicalSpace X₀
    inst✝³ : TopologicalSpace X
    inst✝² : TopologicalSpace Y
    inst✝¹ : TopologicalSpace Z
    inst✝ : LocallyCompactSpace Y
    f : X₀ → X
    hf : Topology.IsQuotientMap f
    g : Prod X Y → Z
    hg : Continuous fun p => g { fst := f p.1, snd := p.2 }
    Gf : ContinuousMap X₀ (ContinuousMap Y Z) := { toFun := fun p => g { fst := f  …
    h : ∀ (x : X), Continuous fun y => g { fst := x, snd := y }
    ⊢ Continuous g
  -/
  let G : X → C(Y, Z) := fun x => ⟨_, h x⟩
  have : Continuous G := by
    rw [hf.continuous_iff]
    exact Gf.continuous
  /-
    X₀ : Type u_1
    X : Type u_2
    Y : Type u_3
    Z : Type u_4
    inst✝⁴ : TopologicalSpace X₀
    inst✝³ : TopologicalSpace X
    inst✝² : TopologicalSpace Y
    inst✝¹ : TopologicalSpace Z
    inst✝ : LocallyCompactSpace Y
    f : X₀ → X
    hf : Topology.IsQuotientMap f
    g : Prod X Y → Z
    hg : Continuous fun p => g { fst := f p.1, snd := p.2 }
    Gf : ContinuousMap X₀ (ContinuousMap Y Z) := { toFun := fun p => g { fst := f  …
    h : ∀ (x : X), Continuous fun y => g { fst := x, snd := y }
    G : X → ContinuousMap Y Z := fun x => { toFun := fun y => g { fst := x, snd := …
    this : Continuous G
    ⊢ Continuous g
  -/
  exact ContinuousMap.continuous_uncurry_of_continuous ⟨G, this⟩
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-22")]
alias QuotientMap.continuous_lift_prod_left := IsQuotientMap.continuous_lift_prod_left


theorem Topology.IsQuotientMap.continuous_lift_prod_right (hf : IsQuotientMap f) {g : Y × X → Z}
    (hg : Continuous fun p : Y × X₀ => g (p.1, f p.2)) : Continuous g := by
  have : Continuous fun p : X₀ × Y => g ((Prod.swap p).1, f (Prod.swap p).2) :=
    hg.comp continuous_swap
  /-
    X₀ : Type u_1
    X : Type u_2
    Y : Type u_3
    Z : Type u_4
    inst✝⁴ : TopologicalSpace X₀
    inst✝³ : TopologicalSpace X
    inst✝² : TopologicalSpace Y
    inst✝¹ : TopologicalSpace Z
    inst✝ : LocallyCompactSpace Y
    f : X₀ → X
    hf : Topology.IsQuotientMap f
    g : Prod Y X → Z
    hg : Continuous fun p => g { fst := p.1, snd := f p.2 }
    this : Continuous fun p => g { fst := p.swap.1, snd := f p.swap.2 }
    ⊢ Continuous g
  -/
  have : Continuous fun p : X₀ × Y => (g ∘ Prod.swap) (f p.1, p.2) := this
  /-
    X₀ : Type u_1
    X : Type u_2
    Y : Type u_3
    Z : Type u_4
    inst✝⁴ : TopologicalSpace X₀
    inst✝³ : TopologicalSpace X
    inst✝² : TopologicalSpace Y
    inst✝¹ : TopologicalSpace Z
    inst✝ : LocallyCompactSpace Y
    f : X₀ → X
    hf : Topology.IsQuotientMap f
    g : Prod Y X → Z
    hg : Continuous fun p => g { fst := p.1, snd := f p.2 }
    this✝ : Continuous fun p => g { fst := p.swap.1, snd := f p.swap.2 }
    this : Continuous fun p => Function.comp g Prod.swap { fst := f p.1, snd := p. …
    ⊢ Continuous g
  -/
  exact (hf.continuous_lift_prod_left this).comp continuous_swap
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-22")]
alias QuotientMap.continuous_lift_prod_right := IsQuotientMap.continuous_lift_prod_right


