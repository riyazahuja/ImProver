/-- The type copy of `E →L[𝕜] F` endowed with the weak operator topology, denoted as
`E →WOT[𝕜] F`. -/
@[irreducible]
def ContinuousLinearMapWOT (𝕜 : Type*) (E : Type*) (F : Type*) [Semiring 𝕜] [AddCommGroup E]
    [TopologicalSpace E] [Module 𝕜 E] [AddCommGroup F] [TopologicalSpace F] [Module 𝕜 F] :=
  E →L[𝕜] F


@[inherit_doc]
notation:25 E " →WOT[" 𝕜 "] " F => ContinuousLinearMapWOT 𝕜 E F


local notation X "⋆" => X →L[𝕜] 𝕜


unseal ContinuousLinearMapWOT in
instance instAddCommGroup [TopologicalAddGroup F] : AddCommGroup (E →WOT[𝕜] F) :=
  inferInstanceAs <| AddCommGroup (E →L[𝕜] F)


unseal ContinuousLinearMapWOT in
instance instModule [TopologicalAddGroup F] [ContinuousConstSMul 𝕜 F] : Module 𝕜 (E →WOT[𝕜] F) :=
  inferInstanceAs <| Module 𝕜 (E →L[𝕜] F)


unseal ContinuousLinearMapWOT in
/-- The linear equivalence that sends a continuous linear map to the type copy endowed with the
weak operator topology. -/
def _root_.ContinuousLinearMap.toWOT :
    (E →L[𝕜] F) ≃ₗ[𝕜] (E →WOT[𝕜] F) :=
  LinearEquiv.refl 𝕜 _


instance instFunLike : FunLike (E →WOT[𝕜] F) E F where
  coe f :=  ((ContinuousLinearMap.toWOT 𝕜 E F).symm f : E → F)
                       /-
                         𝕜 : Type u_1
                         E : Type u_2
                         F : Type u_3
                         inst✝⁸ : NormedField 𝕜
                         inst✝⁷ : AddCommGroup E
                         inst✝⁶ : TopologicalSpace E
                         inst✝⁵ : Module 𝕜 E
                         inst✝⁴ : AddCommGroup F
                         inst✝³ : TopologicalSpace F
                         inst✝² : Module 𝕜 F
                         inst✝¹ : TopologicalAddGroup F
                         inst✝ : ContinuousConstSMul 𝕜 F
                         ⊢ Function.Injective fun f => ⇑((ContinuousLinearMap.toWOT 𝕜 E F).symm f)
                       -/
  coe_injective' := by intro; simp
                              /-
                                🎉 no goals
                              -/


instance instContinuousLinearMapClass : ContinuousLinearMapClass (E →WOT[𝕜] F) 𝕜 E F where
                      /-
                        𝕜 : Type u_1
                        E : Type u_2
                        F : Type u_3
                        inst✝⁸ : NormedField 𝕜
                        inst✝⁷ : AddCommGroup E
                        inst✝⁶ : TopologicalSpace E
                        inst✝⁵ : Module 𝕜 E
                        inst✝⁴ : AddCommGroup F
                        inst✝³ : TopologicalSpace F
                        inst✝² : Module 𝕜 F
                        inst✝¹ : TopologicalAddGroup F
                        inst✝ : ContinuousConstSMul 𝕜 F
                        f : ContinuousLinearMapWOT 𝕜 E F
                        x y : E
                        ⊢ Eq (f (HAdd.hAdd x y)) (HAdd.hAdd (f x) (f y))
                      -/
  map_add f x y := by simp only [DFunLike.coe]; simp
                                                /-
                                                  🎉 no goals
                                                -/
                         /-
                           𝕜 : Type u_1
                           E : Type u_2
                           F : Type u_3
                           inst✝⁸ : NormedField 𝕜
                           inst✝⁷ : AddCommGroup E
                           inst✝⁶ : TopologicalSpace E
                           inst✝⁵ : Module 𝕜 E
                           inst✝⁴ : AddCommGroup F
                           inst✝³ : TopologicalSpace F
                           inst✝² : Module 𝕜 F
                           inst✝¹ : TopologicalAddGroup F
                           inst✝ : ContinuousConstSMul 𝕜 F
                           f : ContinuousLinearMapWOT 𝕜 E F
                           r : 𝕜
                           x : E
                           ⊢ Eq (f (HSMul.hSMul r x)) (HSMul.hSMul ((RingHom.id 𝕜) r) (f x))
                         -/
  map_smulₛₗ f r x := by simp only [DFunLike.coe]; simp
                                                   /-
                                                     🎉 no goals
                                                   -/
  map_continuous f := ContinuousLinearMap.continuous ((ContinuousLinearMap.toWOT 𝕜 E F).symm f)


lemma _root_.ContinuousLinearMap.toWOT_apply {A : E →L[𝕜] F} {x : E} :
    ((ContinuousLinearMap.toWOT 𝕜 E F) A) x = A x := rfl


unseal ContinuousLinearMapWOT in
lemma ext {A B : E →WOT[𝕜] F} (h : ∀ x, A x = B x) : A = B := ContinuousLinearMap.ext h


unseal ContinuousLinearMapWOT in
lemma ext_iff {A B : E →WOT[𝕜] F} : A = B ↔ ∀ x, A x = B x := ContinuousLinearMap.ext_iff

-- This `ext` lemma is set at a lower priority than the default of 1000, so that the
-- version with an inner product (`ContinuousLinearMapWOT.ext_inner`) takes precedence
-- in the case of Hilbert spaces.

@[ext 900]
lemma ext_dual [H : SeparatingDual 𝕜 F] {A B : E →WOT[𝕜] F}
    (h : ∀ x (y : F⋆), y (A x) = y (B x)) : A = B := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁸ : NormedField 𝕜
    inst✝⁷ : AddCommGroup E
    inst✝⁶ : TopologicalSpace E
    inst✝⁵ : Module 𝕜 E
    inst✝⁴ : AddCommGroup F
    inst✝³ : TopologicalSpace F
    inst✝² : Module 𝕜 F
    inst✝¹ : TopologicalAddGroup F
    inst✝ : ContinuousConstSMul 𝕜 F
    H : SeparatingDual 𝕜 F
    A B : ContinuousLinearMapWOT 𝕜 E F
    h : ∀ (x : E) (y : ContinuousLinearMap (RingHom.id 𝕜) F 𝕜), Eq (y (A x)) (y (B …
    ⊢ Eq A B
  -/
  simp_rw [ext_iff, ← (separatingDual_iff_injective.mp H).eq_iff, LinearMap.ext_iff]
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁸ : NormedField 𝕜
    inst✝⁷ : AddCommGroup E
    inst✝⁶ : TopologicalSpace E
    inst✝⁵ : Module 𝕜 E
    inst✝⁴ : AddCommGroup F
    inst✝³ : TopologicalSpace F
    inst✝² : Module 𝕜 F
    inst✝¹ : TopologicalAddGroup F
    inst✝ : ContinuousConstSMul 𝕜 F
    H : SeparatingDual 𝕜 F
    A B : ContinuousLinearMapWOT 𝕜 E F
    h : ∀ (x : E) (y : ContinuousLinearMap (RingHom.id 𝕜) F 𝕜), Eq (y (A x)) (y (B …
    ⊢ ∀ (x : E) (x_1 : ContinuousLinearMap (RingHom.id 𝕜) F 𝕜), Eq (((ContinuousLi …
  -/
  exact h
  /-
    🎉 no goals
  -/


                                                                 /-
                                                                   𝕜 : Type u_1
                                                                   E : Type u_2
                                                                   F : Type u_3
                                                                   inst✝⁸ : NormedField 𝕜
                                                                   inst✝⁷ : AddCommGroup E
                                                                   inst✝⁶ : TopologicalSpace E
                                                                   inst✝⁵ : Module 𝕜 E
                                                                   inst✝⁴ : AddCommGroup F
                                                                   inst✝³ : TopologicalSpace F
                                                                   inst✝² : Module 𝕜 F
                                                                   inst✝¹ : TopologicalAddGroup F
                                                                   inst✝ : ContinuousConstSMul 𝕜 F
                                                                   x : E
                                                                   ⊢ Eq (0 x) 0
                                                                 -/
@[simp] lemma zero_apply (x : E) : (0 : E →WOT[𝕜] F) x = 0 := by simp only [DFunLike.coe]; rfl
                                                                                           /-
                                                                                             🎉 no goals
                                                                                           -/


unseal ContinuousLinearMapWOT in
@[simp] lemma add_apply {f g : E →WOT[𝕜] F} (x : E) : (f + g) x = f x + g x := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁸ : NormedField 𝕜
    inst✝⁷ : AddCommGroup E
    inst✝⁶ : TopologicalSpace E
    inst✝⁵ : Module 𝕜 E
    inst✝⁴ : AddCommGroup F
    inst✝³ : TopologicalSpace F
    inst✝² : Module 𝕜 F
    inst✝¹ : TopologicalAddGroup F
    inst✝ : ContinuousConstSMul 𝕜 F
    f g : ContinuousLinearMapWOT 𝕜 E F
    x : E
    ⊢ Eq ((HAdd.hAdd f g) x) (HAdd.hAdd (f x) (g x))
  -/
  simp only [DFunLike.coe]; rfl
                            /-
                              🎉 no goals
                            -/


unseal ContinuousLinearMapWOT in
@[simp] lemma sub_apply {f g : E →WOT[𝕜] F} (x : E) : (f - g) x = f x - g x := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁸ : NormedField 𝕜
    inst✝⁷ : AddCommGroup E
    inst✝⁶ : TopologicalSpace E
    inst✝⁵ : Module 𝕜 E
    inst✝⁴ : AddCommGroup F
    inst✝³ : TopologicalSpace F
    inst✝² : Module 𝕜 F
    inst✝¹ : TopologicalAddGroup F
    inst✝ : ContinuousConstSMul 𝕜 F
    f g : ContinuousLinearMapWOT 𝕜 E F
    x : E
    ⊢ Eq ((HSub.hSub f g) x) (HSub.hSub (f x) (g x))
  -/
  simp only [DFunLike.coe]; rfl
                            /-
                              🎉 no goals
                            -/


unseal ContinuousLinearMapWOT in
@[simp] lemma neg_apply {f : E →WOT[𝕜] F} (x : E) : (-f) x = -(f x) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁸ : NormedField 𝕜
    inst✝⁷ : AddCommGroup E
    inst✝⁶ : TopologicalSpace E
    inst✝⁵ : Module 𝕜 E
    inst✝⁴ : AddCommGroup F
    inst✝³ : TopologicalSpace F
    inst✝² : Module 𝕜 F
    inst✝¹ : TopologicalAddGroup F
    inst✝ : ContinuousConstSMul 𝕜 F
    f : ContinuousLinearMapWOT 𝕜 E F
    x : E
    ⊢ Eq ((Neg.neg f) x) (Neg.neg (f x))
  -/
  simp only [DFunLike.coe]; rfl
                            /-
                              🎉 no goals
                            -/


unseal ContinuousLinearMapWOT in
@[simp] lemma smul_apply {f : E →WOT[𝕜] F} (c : 𝕜) (x : E) : (c • f) x = c • (f x) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁸ : NormedField 𝕜
    inst✝⁷ : AddCommGroup E
    inst✝⁶ : TopologicalSpace E
    inst✝⁵ : Module 𝕜 E
    inst✝⁴ : AddCommGroup F
    inst✝³ : TopologicalSpace F
    inst✝² : Module 𝕜 F
    inst✝¹ : TopologicalAddGroup F
    inst✝ : ContinuousConstSMul 𝕜 F
    f : ContinuousLinearMapWOT 𝕜 E F
    c : 𝕜
    x : E
    ⊢ Eq ((HSMul.hSMul c f) x) (HSMul.hSMul c (f x))
  -/
  simp only [DFunLike.coe]; rfl
                            /-
                              🎉 no goals
                            -/


variable (𝕜) (E) (F) in
/-- The function that induces the topology on `E →WOT[𝕜] F`, namely the function that takes
an `A` and maps it to `fun ⟨x, y⟩ => y (A x)` in `E × F⋆ → 𝕜`, bundled as a linear map to make
it easier to prove that it is a TVS. -/
def inducingFn : (E →WOT[𝕜] F) →ₗ[𝕜] (E × F⋆ → 𝕜) where
  toFun := fun A ⟨x, y⟩ => y (A x)
                            /-
                              𝕜 : Type u_1
                              E : Type u_2
                              F : Type u_3
                              inst✝⁸ : NormedField 𝕜
                              inst✝⁷ : AddCommGroup E
                              inst✝⁶ : TopologicalSpace E
                              inst✝⁵ : Module 𝕜 E
                              inst✝⁴ : AddCommGroup F
                              inst✝³ : TopologicalSpace F
                              inst✝² : Module 𝕜 F
                              inst✝¹ : TopologicalAddGroup F
                              inst✝ : ContinuousConstSMul 𝕜 F
                              x y : ContinuousLinearMapWOT 𝕜 E F
                              ⊢ Eq ((fun A x => ContinuousLinearMapWOT.inducingFn.match_1 𝕜 E F (fun x => 𝕜) …
                            -/
  map_add' := fun x y => by ext; simp
                                 /-
                                   🎉 no goals
                                 -/
                             /-
                               𝕜 : Type u_1
                               E : Type u_2
                               F : Type u_3
                               inst✝⁸ : NormedField 𝕜
                               inst✝⁷ : AddCommGroup E
                               inst✝⁶ : TopologicalSpace E
                               inst✝⁵ : Module 𝕜 E
                               inst✝⁴ : AddCommGroup F
                               inst✝³ : TopologicalSpace F
                               inst✝² : Module 𝕜 F
                               inst✝¹ : TopologicalAddGroup F
                               inst✝ : ContinuousConstSMul 𝕜 F
                               x : 𝕜
                               y : ContinuousLinearMapWOT 𝕜 E F
                               ⊢ Eq ({ toFun := fun A x => ContinuousLinearMapWOT.inducingFn.match_1 𝕜 E F (f …
                             -/
  map_smul' := fun x y => by ext; simp
                                  /-
                                    🎉 no goals
                                  -/


@[simp]
lemma inducingFn_apply {f : E →WOT[𝕜] F} {x : E} {y : F⋆} :
    inducingFn 𝕜 E F f (x, y) = y (f x) :=
  rfl


/-- The weak operator topology is the coarsest topology such that `fun A => y (A x)` is
continuous for all `x, y`. -/
instance instTopologicalSpace : TopologicalSpace (E →WOT[𝕜] F) :=
  .induced (inducingFn _ _ _) Pi.topologicalSpace


@[fun_prop]
lemma continuous_inducingFn : Continuous (inducingFn 𝕜 E F) :=
  continuous_induced_dom


lemma continuous_dual_apply (x : E) (y : F⋆) : Continuous fun (A : E →WOT[𝕜] F) => y (A x) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁸ : NormedField 𝕜
    inst✝⁷ : AddCommGroup E
    inst✝⁶ : TopologicalSpace E
    inst✝⁵ : Module 𝕜 E
    inst✝⁴ : AddCommGroup F
    inst✝³ : TopologicalSpace F
    inst✝² : Module 𝕜 F
    inst✝¹ : TopologicalAddGroup F
    inst✝ : ContinuousConstSMul 𝕜 F
    x : E
    y : ContinuousLinearMap (RingHom.id 𝕜) F 𝕜
    ⊢ Continuous fun A => y (A x)
  -/
  refine (continuous_pi_iff.mp continuous_inducingFn) ⟨x, y⟩
  /-
    🎉 no goals
  -/


@[fun_prop]
lemma continuous_of_dual_apply_continuous {α : Type*} [TopologicalSpace α] {g : α → E →WOT[𝕜] F}
    (h : ∀ x (y : F⋆), Continuous fun a => y (g a x)) : Continuous g :=
  continuous_induced_rng.2 (continuous_pi_iff.mpr fun p => h p.1 p.2)


lemma isInducing_inducingFn : IsInducing (inducingFn 𝕜 E F) := ⟨rfl⟩


lemma isEmbedding_inducingFn [SeparatingDual 𝕜 F] : IsEmbedding (inducingFn 𝕜 E F) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁹ : NormedField 𝕜
    inst✝⁸ : AddCommGroup E
    inst✝⁷ : TopologicalSpace E
    inst✝⁶ : Module 𝕜 E
    inst✝⁵ : AddCommGroup F
    inst✝⁴ : TopologicalSpace F
    inst✝³ : Module 𝕜 F
    inst✝² : TopologicalAddGroup F
    inst✝¹ : ContinuousConstSMul 𝕜 F
    inst✝ : SeparatingDual 𝕜 F
    ⊢ Topology.IsEmbedding ⇑(ContinuousLinearMapWOT.inducingFn 𝕜 E F)
  -/
  refine Function.Injective.isEmbedding_induced fun A B hAB => ?_
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁹ : NormedField 𝕜
    inst✝⁸ : AddCommGroup E
    inst✝⁷ : TopologicalSpace E
    inst✝⁶ : Module 𝕜 E
    inst✝⁵ : AddCommGroup F
    inst✝⁴ : TopologicalSpace F
    inst✝³ : Module 𝕜 F
    inst✝² : TopologicalAddGroup F
    inst✝¹ : ContinuousConstSMul 𝕜 F
    inst✝ : SeparatingDual 𝕜 F
    A B : ContinuousLinearMapWOT 𝕜 E F
    hAB : Eq ((ContinuousLinearMapWOT.inducingFn 𝕜 E F) A) ((ContinuousLinearMapWO …
    ⊢ Eq A B
  -/
  rw [ContinuousLinearMapWOT.ext_dual_iff]
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁹ : NormedField 𝕜
    inst✝⁸ : AddCommGroup E
    inst✝⁷ : TopologicalSpace E
    inst✝⁶ : Module 𝕜 E
    inst✝⁵ : AddCommGroup F
    inst✝⁴ : TopologicalSpace F
    inst✝³ : Module 𝕜 F
    inst✝² : TopologicalAddGroup F
    inst✝¹ : ContinuousConstSMul 𝕜 F
    inst✝ : SeparatingDual 𝕜 F
    A B : ContinuousLinearMapWOT 𝕜 E F
    hAB : Eq ((ContinuousLinearMapWOT.inducingFn 𝕜 E F) A) ((ContinuousLinearMapWO …
    ⊢ ∀ (x : E) (y : ContinuousLinearMap (RingHom.id 𝕜) F 𝕜), Eq (y (A x)) (y (B x))
  -/
  simpa [funext_iff] using hAB
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-26")]
alias embedding_inducingFn := isEmbedding_inducingFn


open Filter in
/-- The defining property of the weak operator topology: a function `f` tends to
`A : E →WOT[𝕜] F` along filter `l` iff `y (f a x)` tends to `y (A x)` along the same filter. -/
lemma tendsto_iff_forall_dual_apply_tendsto {α : Type*} {l : Filter α} {f : α → E →WOT[𝕜] F}
    {A : E →WOT[𝕜] F} :
    Tendsto f l (𝓝 A) ↔ ∀ x (y : F⋆), Tendsto (fun a => y (f a x)) l (𝓝 (y (A x))) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁸ : NormedField 𝕜
    inst✝⁷ : AddCommGroup E
    inst✝⁶ : TopologicalSpace E
    inst✝⁵ : Module 𝕜 E
    inst✝⁴ : AddCommGroup F
    inst✝³ : TopologicalSpace F
    inst✝² : Module 𝕜 F
    inst✝¹ : TopologicalAddGroup F
    inst✝ : ContinuousConstSMul 𝕜 F
    α : Type u_4
    l : Filter α
    f : α → ContinuousLinearMapWOT 𝕜 E F
    A : ContinuousLinearMapWOT 𝕜 E F
    ⊢ Iff (Filter.Tendsto f l (nhds A)) (∀ (x : E) (y : ContinuousLinearMap (RingH …
  -/
  simp [isInducing_inducingFn.tendsto_nhds_iff, tendsto_pi_nhds]
  /-
    🎉 no goals
  -/


lemma le_nhds_iff_forall_dual_apply_le_nhds {l : Filter (E →WOT[𝕜] F)} {A : E →WOT[𝕜] F} :
    l ≤ 𝓝 A ↔ ∀ x (y : F⋆), l.map (fun T => y (T x)) ≤ 𝓝 (y (A x)) :=
  tendsto_iff_forall_dual_apply_tendsto (f := id)


instance instT3Space [SeparatingDual 𝕜 F] : T3Space (E →WOT[𝕜] F) := isEmbedding_inducingFn.t3Space


instance instContinuousAdd : ContinuousAdd (E →WOT[𝕜] F) := .induced (inducingFn 𝕜 E F)

instance instContinuousNeg : ContinuousNeg (E →WOT[𝕜] F) := .induced (inducingFn 𝕜 E F)

instance instContinuousSMul : ContinuousSMul 𝕜 (E →WOT[𝕜] F) := .induced (inducingFn 𝕜 E F)


instance instTopologicalAddGroup : TopologicalAddGroup (E →WOT[𝕜] F) where


instance instUniformSpace : UniformSpace (E →WOT[𝕜] F) := .comap (inducingFn 𝕜 E F) inferInstance


instance instUniformAddGroup : UniformAddGroup (E →WOT[𝕜] F) := .comap (inducingFn 𝕜 E F)


/-- The family of seminorms that induce the weak operator topology, namely `‖y (A x)‖` for
all `x` and `y`. -/
def seminorm (x : E) (y : F⋆) : Seminorm 𝕜 (E →WOT[𝕜] F) where
  toFun A := ‖y (A x)‖
                  /-
                    𝕜 : Type u_1
                    E : Type u_2
                    F : Type u_3
                    inst✝⁸ : NormedField 𝕜
                    inst✝⁷ : AddCommGroup E
                    inst✝⁶ : TopologicalSpace E
                    inst✝⁵ : Module 𝕜 E
                    inst✝⁴ : AddCommGroup F
                    inst✝³ : TopologicalSpace F
                    inst✝² : Module 𝕜 F
                    inst✝¹ : TopologicalAddGroup F
                    inst✝ : ContinuousConstSMul 𝕜 F
                    x : E
                    y : ContinuousLinearMap (RingHom.id 𝕜) F 𝕜
                    ⊢ Eq ((fun A => Norm.norm (y (A x))) 0) 0
                  -/
  map_zero' := by simp
                  /-
                    🎉 no goals
                  -/
                    /-
                      𝕜 : Type u_1
                      E : Type u_2
                      F : Type u_3
                      inst✝⁸ : NormedField 𝕜
                      inst✝⁷ : AddCommGroup E
                      inst✝⁶ : TopologicalSpace E
                      inst✝⁵ : Module 𝕜 E
                      inst✝⁴ : AddCommGroup F
                      inst✝³ : TopologicalSpace F
                      inst✝² : Module 𝕜 F
                      inst✝¹ : TopologicalAddGroup F
                      inst✝ : ContinuousConstSMul 𝕜 F
                      x : E
                      y : ContinuousLinearMap (RingHom.id 𝕜) F 𝕜
                      A B : ContinuousLinearMapWOT 𝕜 E F
                      ⊢ LE.le ((fun A => Norm.norm (y (A x))) (HAdd.hAdd A B)) (HAdd.hAdd ((fun A => …
                    -/
  add_le' A B := by simpa using norm_add_le _ _
                    /-
                      🎉 no goals
                    -/
               /-
                 𝕜 : Type u_1
                 E : Type u_2
                 F : Type u_3
                 inst✝⁸ : NormedField 𝕜
                 inst✝⁷ : AddCommGroup E
                 inst✝⁶ : TopologicalSpace E
                 inst✝⁵ : Module 𝕜 E
                 inst✝⁴ : AddCommGroup F
                 inst✝³ : TopologicalSpace F
                 inst✝² : Module 𝕜 F
                 inst✝¹ : TopologicalAddGroup F
                 inst✝ : ContinuousConstSMul 𝕜 F
                 x : E
                 y : ContinuousLinearMap (RingHom.id 𝕜) F 𝕜
                 A : ContinuousLinearMapWOT 𝕜 E F
                 ⊢ Eq ((fun A => Norm.norm (y (A x))) (Neg.neg A)) ((fun A => Norm.norm (y (A x …
               -/
  neg' A := by simp
               /-
                 🎉 no goals
               -/
                  /-
                    𝕜 : Type u_1
                    E : Type u_2
                    F : Type u_3
                    inst✝⁸ : NormedField 𝕜
                    inst✝⁷ : AddCommGroup E
                    inst✝⁶ : TopologicalSpace E
                    inst✝⁵ : Module 𝕜 E
                    inst✝⁴ : AddCommGroup F
                    inst✝³ : TopologicalSpace F
                    inst✝² : Module 𝕜 F
                    inst✝¹ : TopologicalAddGroup F
                    inst✝ : ContinuousConstSMul 𝕜 F
                    x : E
                    y : ContinuousLinearMap (RingHom.id 𝕜) F 𝕜
                    r : 𝕜
                    A : ContinuousLinearMapWOT 𝕜 E F
                    ⊢ Eq ({ toFun := fun A => Norm.norm (y (A x)), map_zero' := ⋯, add_le' := ⋯, n …
                  -/
  smul' r A := by simp
                  /-
                    🎉 no goals
                  -/


variable (𝕜) (E) (F) in
/-- The family of seminorms that induce the weak operator topology, namely `‖y (A x)‖` for
all `x` and `y`. -/
def seminormFamily : SeminormFamily 𝕜 (E →WOT[𝕜] F) (E × F⋆) :=
  fun ⟨x, y⟩ => seminorm x y


lemma withSeminorms : WithSeminorms (seminormFamily 𝕜 E F) :=
  let e : E × F⋆ ≃ (Σ _ : E × F⋆, Fin 1) := .symm <| .sigmaUnique _ _
  have : Nonempty (Σ _ : E × F⋆, Fin 1) := e.symm.nonempty
  isInducing_inducingFn.withSeminorms <| withSeminorms_pi (fun _ ↦ norm_withSeminorms 𝕜 𝕜)
    |>.congr_equiv e


lemma hasBasis_seminorms : (𝓝 (0 : E →WOT[𝕜] F)).HasBasis (seminormFamily 𝕜 E F).basisSets id :=
  withSeminorms.hasBasis


instance instLocallyConvexSpace [NormedSpace ℝ 𝕜] [Module ℝ (E →WOT[𝕜] F)]
    [IsScalarTower ℝ 𝕜 (E →WOT[𝕜] F)] :
    LocallyConvexSpace ℝ (E →WOT[𝕜] F) :=
  withSeminorms.toLocallyConvexSpace


/-- The weak operator topology is coarser than the bounded convergence topology, i.e. the inclusion
map is continuous. -/
@[continuity, fun_prop]
lemma ContinuousLinearMap.continuous_toWOT :
    Continuous (ContinuousLinearMap.toWOT 𝕜 E F) :=
  ContinuousLinearMapWOT.continuous_of_dual_apply_continuous fun x y ↦
    y.cont.comp <| continuous_eval_const x


/-- The inclusion map from `E →[𝕜] F` to `E →WOT[𝕜] F`, bundled as a continuous linear map. -/
def ContinuousLinearMap.toWOTCLM : (E →L[𝕜] F) →L[𝕜] (E →WOT[𝕜] F) :=
  ⟨LinearEquiv.toLinearMap (ContinuousLinearMap.toWOT 𝕜 E F), ContinuousLinearMap.continuous_toWOT⟩


