/-- `C_c(α, β)` is the type of continuous functions `α → β` with compact support from a topological
space to a topological space with a zero element.

When possible, instead of parametrizing results over `f : C_c(α, β)`,
you should parametrize over `{F : Type*} [CompactlySupportedContinuousMapClass F α β] (f : F)`.

When you extend this structure, make sure to extend `CompactlySupportedContinuousMapClass`. -/
structure CompactlySupportedContinuousMap (α β : Type*) [TopologicalSpace α] [Zero β]
    [TopologicalSpace β] extends ContinuousMap α β where
  /-- The function has compact support . -/
  hasCompactSupport' : HasCompactSupport toFun


@[inherit_doc]
scoped[CompactlySupported] notation (priority := 2000)
  "C_c(" α ", " β ")" => CompactlySupportedContinuousMap α β


@[inherit_doc]
scoped[CompactlySupported] notation α " →C_c " β => CompactlySupportedContinuousMap α β


/-- `CompactlySupportedContinuousMapClass F α β` states that `F` is a type of continuous maps with
compact support.

You should also extend this typeclass when you extend `CompactlySupportedContinuousMap`. -/
class CompactlySupportedContinuousMapClass (F : Type*) (α β : outParam <| Type*)
    [TopologicalSpace α] [Zero β] [TopologicalSpace β] [FunLike F α β]
    extends ContinuousMapClass F α β : Prop where
  /-- Each member of the class has compact support. -/
  hasCompactSupport (f : F) : HasCompactSupport f


instance : FunLike C_c(α, β) α β where
  coe f := f.toFun
  coe_injective' f g h := by
    /-
      F : Type u_1
      α : Type u_2
      β : Type u_3
      γ : Type u_4
      inst✝² : TopologicalSpace α
      inst✝¹ : TopologicalSpace β
      inst✝ : Zero β
      f g : CompactlySupportedContinuousMap α β
      h : Eq ((fun f => f.toFun) f) ((fun f => f.toFun) g)
      ⊢ Eq f g
    -/
    obtain ⟨⟨_, _⟩, _⟩ := f
    /-
      case mk.mk
      F : Type u_1
      α : Type u_2
      β : Type u_3
      γ : Type u_4
      inst✝² : TopologicalSpace α
      inst✝¹ : TopologicalSpace β
      inst✝ : Zero β
      g : CompactlySupportedContinuousMap α β
      toFun✝ : α → β
      continuous_toFun✝ : Continuous toFun✝
      hasCompactSupport'✝ : HasCompactSupport { toFun := toFun✝, continuous_toFun := …
      h : Eq ((fun f => f.toFun) { toFun := toFun✝, continuous_toFun := continuous_t …
      ⊢ Eq { toFun := toFun✝, continuous_toFun := continuous_toFun✝, hasCompactSuppo …
    -/
    obtain ⟨⟨_, _⟩, _⟩ := g
    /-
      case mk.mk.mk.mk
      F : Type u_1
      α : Type u_2
      β : Type u_3
      γ : Type u_4
      inst✝² : TopologicalSpace α
      inst✝¹ : TopologicalSpace β
      inst✝ : Zero β
      toFun✝¹ : α → β
      continuous_toFun✝¹ : Continuous toFun✝¹
      hasCompactSupport'✝¹ : HasCompactSupport { toFun := toFun✝¹, continuous_toFun  …
      toFun✝ : α → β
      continuous_toFun✝ : Continuous toFun✝
      hasCompactSupport'✝ : HasCompactSupport { toFun := toFun✝, continuous_toFun := …
      h : Eq ((fun f => f.toFun) { toFun := toFun✝¹, continuous_toFun := continuous_ …
      ⊢ Eq { toFun := toFun✝¹, continuous_toFun := continuous_toFun✝¹, hasCompactSup …
    -/
    congr
    /-
      🎉 no goals
    -/


protected lemma hasCompactSupport (f : C_c(α, β)) : HasCompactSupport f := f.hasCompactSupport'


instance : CompactlySupportedContinuousMapClass C_c(α, β) α β where
  map_continuous f := f.continuous_toFun
  hasCompactSupport f := f.hasCompactSupport'


@[simp]
theorem coe_toContinuousMap (f : C_c(α, β)) : (f.toContinuousMap : α → β) = f :=
  rfl


@[ext]
theorem ext {f g : C_c(α, β)} (h : ∀ x, f x = g x) : f = g :=
  DFunLike.ext _ _ h


@[simp]
theorem coe_mk (f : C(α, β)) (h : HasCompactSupport f) : ⇑(⟨f, h⟩ : C_c(α, β)) = f :=
  rfl


/-- Copy of a `CompactlySupportedContinuousMap` with a new `toFun` equal to the old one. Useful
to fix definitional equalities. -/
protected def copy (f : C_c(α, β)) (f' : α → β) (h : f' = f) : C_c(α, β) where
  toFun := f'
  continuous_toFun := by
    /-
      F : Type u_1
      α : Type u_2
      β : Type u_3
      γ : Type u_4
      inst✝² : TopologicalSpace α
      inst✝¹ : TopologicalSpace β
      inst✝ : Zero β
      f : CompactlySupportedContinuousMap α β
      f' : α → β
      h : Eq f' ⇑f
      ⊢ Continuous f'
    -/
    rw [h]
    /-
      F : Type u_1
      α : Type u_2
      β : Type u_3
      γ : Type u_4
      inst✝² : TopologicalSpace α
      inst✝¹ : TopologicalSpace β
      inst✝ : Zero β
      f : CompactlySupportedContinuousMap α β
      f' : α → β
      h : Eq f' ⇑f
      ⊢ Continuous ⇑f
    -/
    exact f.continuous_toFun
    /-
      🎉 no goals
    -/
  hasCompactSupport' := by
    /-
      F : Type u_1
      α : Type u_2
      β : Type u_3
      γ : Type u_4
      inst✝² : TopologicalSpace α
      inst✝¹ : TopologicalSpace β
      inst✝ : Zero β
      f : CompactlySupportedContinuousMap α β
      f' : α → β
      h : Eq f' ⇑f
      ⊢ HasCompactSupport { toFun := f', continuous_toFun := ⋯ }.toFun
    -/
    simp_rw [h]
    /-
      F : Type u_1
      α : Type u_2
      β : Type u_3
      γ : Type u_4
      inst✝² : TopologicalSpace α
      inst✝¹ : TopologicalSpace β
      inst✝ : Zero β
      f : CompactlySupportedContinuousMap α β
      f' : α → β
      h : Eq f' ⇑f
      ⊢ HasCompactSupport ⇑f
    -/
    exact f.hasCompactSupport'
    /-
      🎉 no goals
    -/


@[simp]
theorem coe_copy (f : C_c(α, β)) (f' : α → β) (h : f' = f) : ⇑(f.copy f' h) = f' :=
  rfl


theorem copy_eq (f : C_c(α, β)) (f' : α → β) (h : f' = f) : f.copy f' h = f :=
  DFunLike.ext' h


theorem eq_of_empty [IsEmpty α] (f g : C_c(α, β)) : f = g :=
  ext <| IsEmpty.elim ‹_›


/-- A continuous function on a compact space automatically has compact support. -/
@[simps]
def ContinuousMap.liftCompactlySupported [CompactSpace α] : C(α, β) ≃ C_c(α, β) where
  toFun f :=
    { toFun := f
      hasCompactSupport' := HasCompactSupport.of_compactSpace f }
  invFun f := f
  left_inv _ := rfl
  right_inv _ := rfl


instance [Zero β] : Zero C_c(α, β) where
  zero := { toFun := (0 : C(α, β))
            continuous_toFun := (0 : C(α, β)).2
                                     /-
                                       F : Type u_1
                                       α : Type u_2
                                       β : Type u_3
                                       γ : Type u_4
                                       inst✝² : TopologicalSpace α
                                       inst✝¹ : TopologicalSpace β
                                       x : α
                                       inst✝ : Zero β
                                       ⊢ HasCompactSupport { toFun := ⇑0, continuous_toFun := ⋯ }.toFun
                                     -/
            hasCompactSupport' := by simp [HasCompactSupport, tsupport] }
                                     /-
                                       🎉 no goals
                                     -/


instance [Zero β] : Inhabited C_c(α, β) :=
  ⟨0⟩


@[simp]
theorem coe_zero [Zero β] : ⇑(0 : C_c(α, β)) = 0 :=
  rfl


theorem zero_apply [Zero β] : (0 : C_c(α, β)) x = 0 :=
  rfl


instance [MulZeroClass β] [ContinuousMul β] : Mul C_c(α, β) :=
  ⟨fun f g => ⟨f * g, HasCompactSupport.mul_left g.2⟩⟩


@[simp]
theorem coe_mul [MulZeroClass β] [ContinuousMul β] (f g : C_c(α, β)) : ⇑(f * g) = f * g :=
  rfl


theorem mul_apply [MulZeroClass β] [ContinuousMul β] (f g : C_c(α, β)) : (f * g) x = f x * g x :=
  rfl


/-- the product of `f : F` assuming `ContinuousMapClass F α γ` and `ContinuousSMul γ β` and
`g : C_c(α, β)` is in `C_c(α, β)` -/
instance [Zero β] [TopologicalSpace γ] [SMulZeroClass γ β] [ContinuousSMul γ β]
    {F : Type*} [FunLike F α γ] [ContinuousMapClass F α γ] : SMul F C_c(α, β) where
  smul f g :=
    ⟨⟨fun x ↦ f x • g x, (map_continuous f).smul (map_continuous g)⟩, g.hasCompactSupport.smul_left⟩


@[simp]
theorem coe_smulc [Zero β] [TopologicalSpace γ] [SMulZeroClass γ β] [ContinuousSMul γ β]
    {F : Type*} [FunLike F α γ] [ContinuousMapClass F α γ] (f : F) (g : C_c(α, β)) :
    ⇑(f • g) = fun x => f x • g x :=
  rfl


theorem smulc_apply [Zero β] [TopologicalSpace γ] [SMulZeroClass γ β] [ContinuousSMul γ β]
    {F : Type*} [FunLike F α γ] [ContinuousMapClass F α γ] (f : F) (g : C_c(α, β)) (x : α) :
    (f • g) x = f x • g x :=
  rfl


instance [MulZeroClass β] [ContinuousMul β] : MulZeroClass C_c(α, β) :=
  DFunLike.coe_injective.mulZeroClass _ coe_zero coe_mul


instance [SemigroupWithZero β] [ContinuousMul β] :
    SemigroupWithZero C_c(α, β) :=
  DFunLike.coe_injective.semigroupWithZero _ coe_zero coe_mul


instance [AddZeroClass β] [ContinuousAdd β] : Add C_c(α, β) :=
  ⟨fun f g => ⟨f + g, HasCompactSupport.add f.2 g.2⟩⟩


@[simp]
theorem coe_add [AddZeroClass β] [ContinuousAdd β] (f g : C_c(α, β)) : ⇑(f + g) = f + g :=
  rfl


theorem add_apply [AddZeroClass β] [ContinuousAdd β] (f g : C_c(α, β)) : (f + g) x = f x + g x :=
  rfl


instance [AddZeroClass β] [ContinuousAdd β] : AddZeroClass C_c(α, β) :=
  DFunLike.coe_injective.addZeroClass _ coe_zero coe_add


/-- Coercion to a function as a `AddMonoidHom`. Similar to `AddMonoidHom.coeFn`. -/
def coeFnMonoidHom [AddMonoid β] [ContinuousAdd β] : C_c(α, β) →+ α → β where
  toFun f := f
  map_zero' := coe_zero
  map_add' := coe_add


instance [Zero β] {R : Type*} [SMulZeroClass R β] [ContinuousConstSMul R β] :
    SMul R C_c(α, β) :=
  ⟨fun r f => ⟨⟨r • ⇑f, (map_continuous f).const_smul r⟩, HasCompactSupport.smul_left f.2⟩⟩


@[simp, norm_cast]
theorem coe_smul [Zero β] {R : Type*} [SMulZeroClass R β] [ContinuousConstSMul R β] (r : R)
    (f : C_c(α, β)) : ⇑(r • f) = r • ⇑f :=
  rfl


theorem smul_apply [Zero β] {R : Type*} [SMulZeroClass R β] [ContinuousConstSMul R β] (r : R)
    (f : C_c(α, β)) (x : α) : (r • f) x = r • f x :=
  rfl


instance [AddMonoid β] [ContinuousAdd β] : AddMonoid C_c(α, β) :=
  DFunLike.coe_injective.addMonoid _ coe_zero coe_add fun _ _ => rfl


instance [AddCommMonoid β] [ContinuousAdd β] : AddCommMonoid C_c(α, β) :=
  DFunLike.coe_injective.addCommMonoid _ coe_zero coe_add fun _ _ => rfl


@[simp]
theorem coe_sum [AddCommMonoid β] [ContinuousAdd β] {ι : Type*} (s : Finset ι) (f : ι → C_c(α, β)) :
    ⇑(∑ i in s, f i) = ∑ i in s, (f i : α → β) :=
  map_sum coeFnMonoidHom f s


theorem sum_apply [AddCommMonoid β] [ContinuousAdd β] {ι : Type*} (s : Finset ι) (f : ι → C_c(α, β))
                                                        /-
                                                          α : Type u_2
                                                          β : Type u_3
                                                          inst✝³ : TopologicalSpace α
                                                          inst✝² : TopologicalSpace β
                                                          inst✝¹ : AddCommMonoid β
                                                          inst✝ : ContinuousAdd β
                                                          ι : Type u_5
                                                          s : Finset ι
                                                          f : ι → CompactlySupportedContinuousMap α β
                                                          a : α
                                                          ⊢ Eq ((s.sum fun i => f i) a) (s.sum fun i => (f i) a)
                                                        -/
    (a : α) : (∑ i in s, f i) a = ∑ i in s, f i a := by simp
                                                        /-
                                                          🎉 no goals
                                                        -/


instance : Neg C_c(α, β) where
  neg f := { toFun := -f.1
             continuous_toFun := map_continuous (-f.1)
                                      /-
                                        F : Type u_1
                                        α : Type u_2
                                        β : Type u_3
                                        γ : Type u_4
                                        inst✝³ : TopologicalSpace α
                                        inst✝² : TopologicalSpace β
                                        x : α
                                        inst✝¹ : AddGroup β
                                        inst✝ : TopologicalAddGroup β
                                        f✝ g f : CompactlySupportedContinuousMap α β
                                        ⊢ HasCompactSupport { toFun := Neg.neg ⇑f.toContinuousMap, continuous_toFun := …
                                      -/
             hasCompactSupport' := by simpa [HasCompactSupport, tsupport] using f.2 }
                                      /-
                                        🎉 no goals
                                      -/


@[simp]
theorem coe_neg : ⇑(-f) = -f :=
  rfl


theorem neg_apply : (-f) x = -f x :=
  rfl


instance : Sub C_c(α, β) where
  sub f g := { toFun := f.1 - g.1
               continuous_toFun := map_continuous (f.1 - g.1)
               hasCompactSupport' := by
                 /-
                   F : Type u_1
                   α : Type u_2
                   β : Type u_3
                   γ : Type u_4
                   inst✝³ : TopologicalSpace α
                   inst✝² : TopologicalSpace β
                   x : α
                   inst✝¹ : AddGroup β
                   inst✝ : TopologicalAddGroup β
                   f✝ g✝ f g : CompactlySupportedContinuousMap α β
                   ⊢ HasCompactSupport { toFun := HSub.hSub ⇑f.toContinuousMap ⇑g.toContinuousMap …
                 -/
                 simpa [sub_eq_add_neg] using HasCompactSupport.add f.2 (-g).2 }
                 /-
                   🎉 no goals
                 -/


@[simp]
theorem coe_sub : ⇑(f - g) = f - g :=
  rfl


theorem sub_apply : (f - g) x = f x - g x :=
  rfl


instance : AddGroup C_c(α, β) :=
  DFunLike.coe_injective.addGroup _ coe_zero coe_add coe_neg coe_sub (fun _ _ => rfl) fun _ _ => rfl


instance [AddCommGroup β] [TopologicalAddGroup β] : AddCommGroup C_c(α, β) :=
  DFunLike.coe_injective.addCommGroup _ coe_zero coe_add coe_neg coe_sub (fun _ _ => rfl) fun _ _ =>
    rfl


instance [Zero β] {R : Type*} [Zero R] [SMulWithZero R β] [SMulWithZero Rᵐᵒᵖ β]
    [ContinuousConstSMul R β] [IsCentralScalar R β] : IsCentralScalar R C_c(α, β) :=
  ⟨fun _ _ => ext fun _ => op_smul_eq_smul _ _⟩


instance [Zero β] {R : Type*} [Zero R] [SMulWithZero R β]
    [ContinuousConstSMul R β] : SMulWithZero R C_c(α, β) :=
  Function.Injective.smulWithZero ⟨_, coe_zero⟩ DFunLike.coe_injective coe_smul


instance [Zero β] {R : Type*} [MonoidWithZero R] [MulActionWithZero R β]
    [ContinuousConstSMul R β] : MulActionWithZero R C_c(α, β) :=
  Function.Injective.mulActionWithZero ⟨_, coe_zero⟩ DFunLike.coe_injective coe_smul


instance [AddCommMonoid β] [ContinuousAdd β] {R : Type*} [Semiring R] [Module R β]
    [ContinuousConstSMul R β] : Module R C_c(α, β) :=
  Function.Injective.module R ⟨⟨_, coe_zero⟩, coe_add⟩ DFunLike.coe_injective coe_smul


instance [NonUnitalNonAssocSemiring β] [TopologicalSemiring β] :
    NonUnitalNonAssocSemiring C_c(α, β) :=
  DFunLike.coe_injective.nonUnitalNonAssocSemiring _ coe_zero coe_add coe_mul fun _ _ => rfl


instance [NonUnitalSemiring β] [TopologicalSemiring β] :
    NonUnitalSemiring C_c(α, β) :=
  DFunLike.coe_injective.nonUnitalSemiring _ coe_zero coe_add coe_mul fun _ _ => rfl


instance [NonUnitalCommSemiring β] [TopologicalSemiring β] :
    NonUnitalCommSemiring C_c(α, β) :=
  DFunLike.coe_injective.nonUnitalCommSemiring _ coe_zero coe_add coe_mul fun _ _ => rfl


instance [NonUnitalNonAssocRing β] [TopologicalRing β] :
    NonUnitalNonAssocRing C_c(α, β) :=
  DFunLike.coe_injective.nonUnitalNonAssocRing _ coe_zero coe_add coe_mul coe_neg coe_sub
    (fun _ _ => rfl) fun _ _ => rfl


instance [NonUnitalRing β] [TopologicalRing β] : NonUnitalRing C_c(α, β) :=
  DFunLike.coe_injective.nonUnitalRing _ coe_zero coe_add coe_mul coe_neg coe_sub (fun _ _ => rfl)
    fun _ _ => rfl


instance [NonUnitalCommRing β] [TopologicalRing β] :
    NonUnitalCommRing C_c(α, β) :=
  DFunLike.coe_injective.nonUnitalCommRing _ coe_zero coe_add coe_mul coe_neg coe_sub
    (fun _ _ => rfl) fun _ _ => rfl


instance {R : Type*} [Semiring R] [NonUnitalNonAssocSemiring β]
    [TopologicalSemiring β] [Module R β] [ContinuousConstSMul R β] [IsScalarTower R β β] :
    IsScalarTower R C_c(α, β) C_c(α, β) where
  smul_assoc r f g := by
    /-
      F : Type u_1
      α : Type u_2
      β : Type u_3
      γ : Type u_4
      inst✝⁷ : TopologicalSpace α
      inst✝⁶ : TopologicalSpace β
      x : α
      R : Type u_5
      inst✝⁵ : Semiring R
      inst✝⁴ : NonUnitalNonAssocSemiring β
      inst✝³ : TopologicalSemiring β
      inst✝² : Module R β
      inst✝¹ : ContinuousConstSMul R β
      inst✝ : IsScalarTower R β β
      r : R
      f g : CompactlySupportedContinuousMap α β
      ⊢ Eq (HSMul.hSMul (HSMul.hSMul r f) g) (HSMul.hSMul r (HSMul.hSMul f g))
    -/
    ext
    /-
      case h
      F : Type u_1
      α : Type u_2
      β : Type u_3
      γ : Type u_4
      inst✝⁷ : TopologicalSpace α
      inst✝⁶ : TopologicalSpace β
      x : α
      R : Type u_5
      inst✝⁵ : Semiring R
      inst✝⁴ : NonUnitalNonAssocSemiring β
      inst✝³ : TopologicalSemiring β
      inst✝² : Module R β
      inst✝¹ : ContinuousConstSMul R β
      inst✝ : IsScalarTower R β β
      r : R
      f g : CompactlySupportedContinuousMap α β
      x✝ : α
      ⊢ Eq ((HSMul.hSMul (HSMul.hSMul r f) g) x✝) ((HSMul.hSMul r (HSMul.hSMul f g)) …
    -/
    simp only [smul_eq_mul, coe_mul, coe_smul, Pi.mul_apply, Pi.smul_apply]
    /-
      case h
      F : Type u_1
      α : Type u_2
      β : Type u_3
      γ : Type u_4
      inst✝⁷ : TopologicalSpace α
      inst✝⁶ : TopologicalSpace β
      x : α
      R : Type u_5
      inst✝⁵ : Semiring R
      inst✝⁴ : NonUnitalNonAssocSemiring β
      inst✝³ : TopologicalSemiring β
      inst✝² : Module R β
      inst✝¹ : ContinuousConstSMul R β
      inst✝ : IsScalarTower R β β
      r : R
      f g : CompactlySupportedContinuousMap α β
      x✝ : α
      ⊢ Eq (HMul.hMul (HSMul.hSMul r (f x✝)) (g x✝)) (HSMul.hSMul r (HMul.hMul (f x✝ …
    -/
    rw [← smul_eq_mul, ← smul_eq_mul, smul_assoc]
    /-
      🎉 no goals
    -/


instance {R : Type*} [Semiring R] [NonUnitalNonAssocSemiring β]
    [TopologicalSemiring β] [Module R β] [ContinuousConstSMul R β] [SMulCommClass R β β] :
    SMulCommClass R C_c(α, β) C_c(α, β) where
  smul_comm r f g := by
    /-
      F : Type u_1
      α : Type u_2
      β : Type u_3
      γ : Type u_4
      inst✝⁷ : TopologicalSpace α
      inst✝⁶ : TopologicalSpace β
      x : α
      R : Type u_5
      inst✝⁵ : Semiring R
      inst✝⁴ : NonUnitalNonAssocSemiring β
      inst✝³ : TopologicalSemiring β
      inst✝² : Module R β
      inst✝¹ : ContinuousConstSMul R β
      inst✝ : SMulCommClass R β β
      r : R
      f g : CompactlySupportedContinuousMap α β
      ⊢ Eq (HSMul.hSMul r (HSMul.hSMul f g)) (HSMul.hSMul f (HSMul.hSMul r g))
    -/
    ext
    /-
      case h
      F : Type u_1
      α : Type u_2
      β : Type u_3
      γ : Type u_4
      inst✝⁷ : TopologicalSpace α
      inst✝⁶ : TopologicalSpace β
      x : α
      R : Type u_5
      inst✝⁵ : Semiring R
      inst✝⁴ : NonUnitalNonAssocSemiring β
      inst✝³ : TopologicalSemiring β
      inst✝² : Module R β
      inst✝¹ : ContinuousConstSMul R β
      inst✝ : SMulCommClass R β β
      r : R
      f g : CompactlySupportedContinuousMap α β
      x✝ : α
      ⊢ Eq ((HSMul.hSMul r (HSMul.hSMul f g)) x✝) ((HSMul.hSMul f (HSMul.hSMul r g)) …
    -/
    simp only [smul_eq_mul, coe_smul, coe_mul, Pi.smul_apply, Pi.mul_apply]
    /-
      case h
      F : Type u_1
      α : Type u_2
      β : Type u_3
      γ : Type u_4
      inst✝⁷ : TopologicalSpace α
      inst✝⁶ : TopologicalSpace β
      x : α
      R : Type u_5
      inst✝⁵ : Semiring R
      inst✝⁴ : NonUnitalNonAssocSemiring β
      inst✝³ : TopologicalSemiring β
      inst✝² : Module R β
      inst✝¹ : ContinuousConstSMul R β
      inst✝ : SMulCommClass R β β
      r : R
      f g : CompactlySupportedContinuousMap α β
      x✝ : α
      ⊢ Eq (HSMul.hSMul r (HMul.hMul (f x✝) (g x✝))) (HMul.hMul (f x✝) (HSMul.hSMul  …
    -/
    rw [← smul_eq_mul, ← smul_eq_mul, smul_comm]
    /-
      🎉 no goals
    -/


instance : Star C_c(α, β) where
  star f :=
    { toFun := fun x => star (f x)
      continuous_toFun := (map_continuous f).star
      hasCompactSupport' := by
        /-
          F : Type u_1
          α : Type u_2
          β : Type u_3
          γ : Type u_4
          inst✝⁴ : TopologicalSpace α
          inst✝³ : TopologicalSpace β
          inst✝² : AddMonoid β
          inst✝¹ : StarAddMonoid β
          inst✝ : ContinuousStar β
          f : CompactlySupportedContinuousMap α β
          ⊢ HasCompactSupport { toFun := fun x => Star.star (f x), continuous_toFun := ⋯ …
        -/
        rw [HasCompactSupport, tsupport]
        /-
          F : Type u_1
          α : Type u_2
          β : Type u_3
          γ : Type u_4
          inst✝⁴ : TopologicalSpace α
          inst✝³ : TopologicalSpace β
          inst✝² : AddMonoid β
          inst✝¹ : StarAddMonoid β
          inst✝ : ContinuousStar β
          f : CompactlySupportedContinuousMap α β
          ⊢ IsCompact (closure (Function.support { toFun := fun x => Star.star (f x), co …
        -/
        simp only
        have support_star : (Function.support fun (x : α) => star (f x)) = Function.support f := by
          ext x
          simp only [Function.mem_support, ne_eq, star_eq_zero]
        /-
          F : Type u_1
          α : Type u_2
          β : Type u_3
          γ : Type u_4
          inst✝⁴ : TopologicalSpace α
          inst✝³ : TopologicalSpace β
          inst✝² : AddMonoid β
          inst✝¹ : StarAddMonoid β
          inst✝ : ContinuousStar β
          f : CompactlySupportedContinuousMap α β
          support_star : Eq (Function.support fun x => Star.star (f x)) (Function.suppor …
          ⊢ IsCompact (closure (Function.support fun x => Star.star (f x)))
        -/
        rw [support_star]
        /-
          F : Type u_1
          α : Type u_2
          β : Type u_3
          γ : Type u_4
          inst✝⁴ : TopologicalSpace α
          inst✝³ : TopologicalSpace β
          inst✝² : AddMonoid β
          inst✝¹ : StarAddMonoid β
          inst✝ : ContinuousStar β
          f : CompactlySupportedContinuousMap α β
          support_star : Eq (Function.support fun x => Star.star (f x)) (Function.suppor …
          ⊢ IsCompact (closure (Function.support ⇑f))
        -/
        exact f.2 }
        /-
          🎉 no goals
        -/


@[simp]
theorem coe_star (f : C_c(α, β)) : ⇑(star f) = star (⇑f) :=
  rfl


theorem star_apply (f : C_c(α, β)) (x : α) : (star f) x = star (f x) :=
  rfl


instance [TrivialStar β] : TrivialStar C_c(α, β) where
    star_trivial f := ext fun x => star_trivial (f x)


instance [ContinuousAdd β] : StarAddMonoid C_c(α, β) where
  star_involutive f := ext fun x => star_star (f x)
  star_add f g := ext fun x => star_add (f x) (g x)


instance : StarModule 𝕜 C_c(α, β) where
  star_smul k f := ext fun x => star_smul k (f x)


instance : StarRing C_c(α, β) :=
  { CompactlySupportedContinuousMap.instStarAddMonoid with
    star_mul := fun f g => ext fun x => star_mul (f x) (g x) }


local notation α " →co " β => CocompactMap α β


/-- Composition of a continuous function with compact support with a cocompact map
yields another continuous function with compact support. -/
def comp (f : C_c(γ, δ)) (g : β →co γ) : C_c(β, δ) where
  toContinuousMap := (f : C(γ, δ)).comp g
  hasCompactSupport' := by
    apply IsCompact.of_isClosed_subset (g.isCompact_preimage_of_isClosed f.2 (isClosed_tsupport _))
      (isClosed_tsupport (f ∘ g))
    /-
      F : Type u_1
      α : Type u_2
      β : Type u_3
      γ : Type u_4
      inst✝⁴ : TopologicalSpace α
      δ : Type u_5
      inst✝³ : TopologicalSpace β
      inst✝² : TopologicalSpace γ
      inst✝¹ : TopologicalSpace δ
      inst✝ : Zero δ
      f : CompactlySupportedContinuousMap γ δ
      g : CocompactMap β γ
      ⊢ HasSubset.Subset (tsupport (Function.comp ⇑f ⇑g)) (Set.preimage (⇑g) (tsuppo …
    -/
    intro x hx
    /-
      F : Type u_1
      α : Type u_2
      β : Type u_3
      γ : Type u_4
      inst✝⁴ : TopologicalSpace α
      δ : Type u_5
      inst✝³ : TopologicalSpace β
      inst✝² : TopologicalSpace γ
      inst✝¹ : TopologicalSpace δ
      inst✝ : Zero δ
      f : CompactlySupportedContinuousMap γ δ
      g : CocompactMap β γ
      x : β
      hx : Membership.mem (tsupport (Function.comp ⇑f ⇑g)) x
      ⊢ Membership.mem (Set.preimage (⇑g) (tsupport f.toFun)) x
    -/
    rw [tsupport, Set.mem_preimage, _root_.mem_closure_iff]
    /-
      F : Type u_1
      α : Type u_2
      β : Type u_3
      γ : Type u_4
      inst✝⁴ : TopologicalSpace α
      δ : Type u_5
      inst✝³ : TopologicalSpace β
      inst✝² : TopologicalSpace γ
      inst✝¹ : TopologicalSpace δ
      inst✝ : Zero δ
      f : CompactlySupportedContinuousMap γ δ
      g : CocompactMap β γ
      x : β
      hx : Membership.mem (tsupport (Function.comp ⇑f ⇑g)) x
      ⊢ ∀ (o : Set γ), IsOpen o → Membership.mem o (g x) → (Inter.inter o (Function. …
    -/
    intro o ho hgxo
    /-
      F : Type u_1
      α : Type u_2
      β : Type u_3
      γ : Type u_4
      inst✝⁴ : TopologicalSpace α
      δ : Type u_5
      inst✝³ : TopologicalSpace β
      inst✝² : TopologicalSpace γ
      inst✝¹ : TopologicalSpace δ
      inst✝ : Zero δ
      f : CompactlySupportedContinuousMap γ δ
      g : CocompactMap β γ
      x : β
      hx : Membership.mem (tsupport (Function.comp ⇑f ⇑g)) x
      o : Set γ
      ho : IsOpen o
      hgxo : Membership.mem o (g x)
      ⊢ (Inter.inter o (Function.support f.toFun)).Nonempty
    -/
    rw [tsupport, _root_.mem_closure_iff] at hx
    /-
      F : Type u_1
      α : Type u_2
      β : Type u_3
      γ : Type u_4
      inst✝⁴ : TopologicalSpace α
      δ : Type u_5
      inst✝³ : TopologicalSpace β
      inst✝² : TopologicalSpace γ
      inst✝¹ : TopologicalSpace δ
      inst✝ : Zero δ
      f : CompactlySupportedContinuousMap γ δ
      g : CocompactMap β γ
      x : β
      hx : ∀ (o : Set β), IsOpen o → Membership.mem o x → (Inter.inter o (Function.s …
      o : Set γ
      ho : IsOpen o
      hgxo : Membership.mem o (g x)
      ⊢ (Inter.inter o (Function.support f.toFun)).Nonempty
    -/
    obtain ⟨y, hy⟩ := hx (g ⁻¹' o) (IsOpen.preimage g.1.2 ho) hgxo
    /-
      case intro
      F : Type u_1
      α : Type u_2
      β : Type u_3
      γ : Type u_4
      inst✝⁴ : TopologicalSpace α
      δ : Type u_5
      inst✝³ : TopologicalSpace β
      inst✝² : TopologicalSpace γ
      inst✝¹ : TopologicalSpace δ
      inst✝ : Zero δ
      f : CompactlySupportedContinuousMap γ δ
      g : CocompactMap β γ
      x : β
      hx : ∀ (o : Set β), IsOpen o → Membership.mem o x → (Inter.inter o (Function.s …
      o : Set γ
      ho : IsOpen o
      hgxo : Membership.mem o (g x)
      y : β
      hy : Membership.mem (Inter.inter (Set.preimage (⇑g) o) (Function.support (Func …
      ⊢ (Inter.inter o (Function.support f.toFun)).Nonempty
    -/
    exact ⟨g y, hy⟩
    /-
      🎉 no goals
    -/


@[simp]
theorem coe_comp_to_continuous_fun (f : C_c(γ, δ)) (g : β →co γ) : ((f.comp g) : β → δ) = f ∘ g :=
  rfl


@[simp]
theorem comp_id (f : C_c(γ, δ)) : f.comp (CocompactMap.id γ) = f :=
  ext fun _ => rfl


@[simp]
theorem comp_assoc (f : C_c(γ, δ)) (g : β →co γ) (h : α →co β) :
    (f.comp g).comp h = f.comp (g.comp h) :=
  rfl


@[simp]
theorem zero_comp (g : β →co γ) : (0 : C_c(γ, δ)).comp g = 0 :=
  rfl


/-- Composition as an additive monoid homomorphism. -/
def compAddMonoidHom [AddMonoid δ] [ContinuousAdd δ] (g : β →co γ) : C_c(γ, δ) →+ C_c(β, δ) where
  toFun f := f.comp g
  map_zero' := zero_comp g
  map_add' _ _ := rfl


/-- Composition as a semigroup homomorphism. -/
def compMulHom [MulZeroClass δ] [ContinuousMul δ] (g : β →co γ) : C_c(γ, δ) →ₙ* C_c(β, δ) where
  toFun f := f.comp g
  map_mul' _ _ := rfl


/-- Composition as a linear map. -/
def compLinearMap [AddCommMonoid δ] [ContinuousAdd δ] {R : Type*} [Semiring R] [Module R δ]
    [ContinuousConstSMul R δ] (g : β →co γ) : C_c(γ, δ) →ₗ[R] C_c(β, δ) where
  toFun f := f.comp g
  map_add' _ _ := rfl
  map_smul' _ _ := rfl


/-- Composition as a non-unital algebra homomorphism. -/
def compNonUnitalAlgHom {R : Type*} [Semiring R] [NonUnitalNonAssocSemiring δ]
    [TopologicalSemiring δ] [Module R δ] [ContinuousConstSMul R δ] (g : β →co γ) :
    C_c(γ, δ) →ₙₐ[R] C_c(β, δ) where
  toFun f := f.comp g
  map_smul' _ _ := rfl
  map_zero' := rfl
  map_add' _ _ := rfl
  map_mul' _ _ := rfl


instance : CoeTC F (CompactlySupportedContinuousMap α β) :=
  ⟨fun f =>
    { toFun := f
      continuous_toFun := map_continuous f
      hasCompactSupport' := hasCompactSupport f }⟩


/-- A continuous function on a compact space has automatically compact support. This is not an
instance to avoid type class loops. -/
lemma of_compactSpace (G : Type*) [FunLike G α β]
    [ContinuousMapClass G α β] [CompactSpace α] : CompactlySupportedContinuousMapClass G α β where
  map_continuous := map_continuous
  hasCompactSupport := by
    /-
      α : Type u_2
      β : Type u_3
      inst✝⁵ : TopologicalSpace α
      inst✝⁴ : Zero β
      inst✝³ : TopologicalSpace β
      G : Type u_5
      inst✝² : FunLike G α β
      inst✝¹ : ContinuousMapClass G α β
      inst✝ : CompactSpace α
      ⊢ ∀ (f : G), HasCompactSupport ⇑f
    -/
    intro f
    /-
      α : Type u_2
      β : Type u_3
      inst✝⁵ : TopologicalSpace α
      inst✝⁴ : Zero β
      inst✝³ : TopologicalSpace β
      G : Type u_5
      inst✝² : FunLike G α β
      inst✝¹ : ContinuousMapClass G α β
      inst✝ : CompactSpace α
      f : G
      ⊢ HasCompactSupport ⇑f
    -/
    exact HasCompactSupport.of_compactSpace f
    /-
      🎉 no goals
    -/


theorem uniformContinuous (f : F) : UniformContinuous (f : β → γ) :=
  (map_continuous f).uniformContinuous_of_tendsto_cocompact
  (HasCompactSupport.is_zero_at_infty (hasCompactSupport f))


instance : ZeroAtInftyContinuousMapClass F β γ where
  zero_at_infty f := HasCompactSupport.is_zero_at_infty (hasCompactSupport f)


/-- The nonnegative part of a bounded continuous `ℝ`-valued function as a bounded
continuous `ℝ≥0`-valued function. -/
noncomputable def nnrealPart (f : C_c(α, ℝ)) : C_c(α, ℝ≥0) where
  toFun := Real.toNNReal.comp f.toFun
  continuous_toFun := Continuous.comp continuous_real_toNNReal f.continuous
  hasCompactSupport' := by
    /-
      F : Type u_1
      α : Type u_2
      β : Type u_3
      γ : Type u_4
      inst✝ : TopologicalSpace α
      f : CompactlySupportedContinuousMap α Real
      ⊢ HasCompactSupport { toFun := Function.comp Real.toNNReal f.toFun, continuous …
    -/
    apply HasCompactSupport.comp_left f.hasCompactSupport' Real.toNNReal_zero
    /-
      🎉 no goals
    -/


@[simp]
lemma nnrealPart_apply (f : C_c(α, ℝ)) (x : α) :
    f.nnrealPart x = Real.toNNReal (f x) := rfl


