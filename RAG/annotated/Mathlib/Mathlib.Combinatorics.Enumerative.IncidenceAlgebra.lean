/-- The `𝕜`-incidence algebra over `α`. -/
structure IncidenceAlgebra (𝕜 α : Type*) [Zero 𝕜] [LE α] where
  /-- The underlying function of an element of the incidence algebra.

  Do not use this function directly. Instead use the coercion coming from the `FunLike`
  instance. -/
  toFun : α → α → 𝕜
  eq_zero_of_not_le' ⦃a b : α⦄ : ¬a ≤ b → toFun a b = 0


instance instFunLike : FunLike (IncidenceAlgebra 𝕜 α) α (α → 𝕜) where
  coe := toFun
                             /-
                               F : Type u_1
                               𝕜 : Type u_2
                               𝕝 : Type u_3
                               𝕞 : Type u_4
                               α : Type u_5
                               β : Type u_6
                               inst✝¹ : Zero 𝕜
                               inst✝ : LE α
                               a b : α
                               f g : IncidenceAlgebra 𝕜 α
                               h : Eq f.toFun g.toFun
                               ⊢ Eq f g
                             -/
  coe_injective' f g h := by cases f; cases g; congr
                                               /-
                                                 🎉 no goals
                                               -/


lemma apply_eq_zero_of_not_le (h : ¬a ≤ b) (f : IncidenceAlgebra 𝕜 α) : f a b = 0 :=
  eq_zero_of_not_le' _ h


lemma le_of_ne_zero {f : IncidenceAlgebra 𝕜 α} : f a b ≠ 0 → a ≤ b :=
  not_imp_comm.1 fun h ↦ apply_eq_zero_of_not_le h _


@[simp] lemma toFun_eq_coe (f : IncidenceAlgebra 𝕜 α) : f.toFun = f := rfl

@[simp, norm_cast] lemma coe_mk (f : α → α → 𝕜) (h) : (mk f h : α → α → 𝕜) = f := rfl


lemma coe_inj {f g : IncidenceAlgebra 𝕜 α} : (f : α → α → 𝕜) = g ↔ f = g :=
  DFunLike.coe_injective.eq_iff


@[ext]
lemma ext ⦃f g : IncidenceAlgebra 𝕜 α⦄ (h : ∀ a b, a ≤ b → f a b = g a b) : f = g := by
  /-
    𝕜 : Type u_2
    α : Type u_5
    inst✝¹ : Zero 𝕜
    inst✝ : LE α
    f g : IncidenceAlgebra 𝕜 α
    h : ∀ (a b : α), LE.le a b → Eq (f a b) (g a b)
    ⊢ Eq f g
  -/
  refine DFunLike.coe_injective' (funext₂ fun a b ↦ ?_)
  /-
    𝕜 : Type u_2
    α : Type u_5
    inst✝¹ : Zero 𝕜
    inst✝ : LE α
    f g : IncidenceAlgebra 𝕜 α
    h : ∀ (a b : α), LE.le a b → Eq (f a b) (g a b)
    a b : α
    ⊢ Eq (f a b) (g a b)
  -/
  by_cases hab : a ≤ b
    /-
      case pos
      𝕜 : Type u_2
      α : Type u_5
      inst✝¹ : Zero 𝕜
      inst✝ : LE α
      f g : IncidenceAlgebra 𝕜 α
      h : ∀ (a b : α), LE.le a b → Eq (f a b) (g a b)
      a b : α
      hab : LE.le a b
      ⊢ Eq (f a b) (g a b)
    -/
  · exact h _ _ hab
    /-
      🎉 no goals
    -/
    /-
      case neg
      𝕜 : Type u_2
      α : Type u_5
      inst✝¹ : Zero 𝕜
      inst✝ : LE α
      f g : IncidenceAlgebra 𝕜 α
      h : ∀ (a b : α), LE.le a b → Eq (f a b) (g a b)
      a b : α
      hab : Not (LE.le a b)
      ⊢ Eq (f a b) (g a b)
    -/
  · rw [apply_eq_zero_of_not_le hab, apply_eq_zero_of_not_le hab]
    /-
      🎉 no goals
    -/


@[simp] lemma mk_coe (f : IncidenceAlgebra 𝕜 α) (h) : mk f h = f := rfl


instance instZero : Zero (IncidenceAlgebra 𝕜 α) := ⟨⟨fun _ _ ↦ 0, fun _ _ _ ↦ rfl⟩⟩

instance instInhabited : Inhabited (IncidenceAlgebra 𝕜 α) := ⟨0⟩


@[simp, norm_cast] lemma coe_zero : ⇑(0 : IncidenceAlgebra 𝕜 α) = 0 := rfl

lemma zero_apply (a b : α) : (0 : IncidenceAlgebra 𝕜 α) a b = 0 := rfl


instance instAdd : Add (IncidenceAlgebra 𝕜 α) where
                                    /-
                                      F : Type u_1
                                      𝕜 : Type u_2
                                      𝕝 : Type u_3
                                      𝕞 : Type u_4
                                      α : Type u_5
                                      β : Type u_6
                                      inst✝¹ : AddZeroClass 𝕜
                                      inst✝ : LE α
                                      f g : IncidenceAlgebra 𝕜 α
                                      a b : α
                                      h : Not (LE.le a b)
                                      ⊢ Eq (HAdd.hAdd (⇑f) (⇑g) a b) 0
                                    -/
  add f g := ⟨f + g, fun a b h ↦ by simp_rw [Pi.add_apply, apply_eq_zero_of_not_le h, zero_add]⟩
                                    /-
                                      🎉 no goals
                                    -/


@[simp, norm_cast] lemma coe_add (f g : IncidenceAlgebra 𝕜 α) : ⇑(f + g) = f + g := rfl

lemma add_apply (f g : IncidenceAlgebra 𝕜 α) (a b : α) : (f + g) a b = f a b + g a b := rfl


instance instSmulZeroClassRight : SMulZeroClass M (IncidenceAlgebra 𝕜 α) where
  smul c f :=
                              /-
                                F : Type u_1
                                𝕜 : Type u_2
                                𝕝 : Type u_3
                                𝕞 : Type u_4
                                α : Type u_5
                                β : Type u_6
                                M : Type u_7
                                inst✝² : Zero 𝕜
                                inst✝¹ : LE α
                                inst✝ : SMulZeroClass M 𝕜
                                c : M
                                f : IncidenceAlgebra 𝕜 α
                                a b : α
                                hab : Not (LE.le a b)
                                ⊢ Eq (HSMul.hSMul c (⇑f) a b) 0
                              -/
    ⟨c • ⇑f, fun a b hab ↦ by simp_rw [Pi.smul_apply, apply_eq_zero_of_not_le hab, smul_zero]⟩
                              /-
                                🎉 no goals
                              -/
                    /-
                      F : Type u_1
                      𝕜 : Type u_2
                      𝕝 : Type u_3
                      𝕞 : Type u_4
                      α : Type u_5
                      β : Type u_6
                      M : Type u_7
                      inst✝² : Zero 𝕜
                      inst✝¹ : LE α
                      inst✝ : SMulZeroClass M 𝕜
                      c : M
                      ⊢ Eq (HSMul.hSMul c 0) 0
                    -/
  smul_zero c := by ext; exact smul_zero _
                         /-
                           🎉 no goals
                         -/


@[simp, norm_cast] lemma coe_constSMul (c : M) (f : IncidenceAlgebra 𝕜 α) : ⇑(c • f) = c • ⇑f := rfl


lemma constSMul_apply (c : M) (f : IncidenceAlgebra 𝕜 α) (a b : α) : (c • f) a b = c • f a b := rfl


instance instAddMonoid [AddMonoid 𝕜] [LE α] : AddMonoid (IncidenceAlgebra 𝕜 α) :=
  DFunLike.coe_injective.addMonoid _ coe_zero coe_add fun _ _ ↦ rfl


instance instAddCommMonoid [AddCommMonoid 𝕜] [LE α] : AddCommMonoid (IncidenceAlgebra 𝕜 α) :=
  DFunLike.coe_injective.addCommMonoid _ coe_zero coe_add fun _ _ ↦ rfl


instance instNeg : Neg (IncidenceAlgebra 𝕜 α) where
                               /-
                                 F : Type u_1
                                 𝕜 : Type u_2
                                 𝕝 : Type u_3
                                 𝕞 : Type u_4
                                 α : Type u_5
                                 β : Type u_6
                                 inst✝¹ : AddGroup 𝕜
                                 inst✝ : LE α
                                 f : IncidenceAlgebra 𝕜 α
                                 a b : α
                                 h : Not (LE.le a b)
                                 ⊢ Eq (Neg.neg (⇑f) a b) 0
                               -/
  neg f := ⟨-f, fun a b h ↦ by simp_rw [Pi.neg_apply, apply_eq_zero_of_not_le h, neg_zero]⟩
                               /-
                                 🎉 no goals
                               -/


instance instSub : Sub (IncidenceAlgebra 𝕜 α) where
                                    /-
                                      F : Type u_1
                                      𝕜 : Type u_2
                                      𝕝 : Type u_3
                                      𝕞 : Type u_4
                                      α : Type u_5
                                      β : Type u_6
                                      inst✝¹ : AddGroup 𝕜
                                      inst✝ : LE α
                                      f g : IncidenceAlgebra 𝕜 α
                                      a b : α
                                      h : Not (LE.le a b)
                                      ⊢ Eq (HSub.hSub (⇑f) (⇑g) a b) 0
                                    -/
  sub f g := ⟨f - g, fun a b h ↦ by simp_rw [Pi.sub_apply, apply_eq_zero_of_not_le h, sub_zero]⟩
                                    /-
                                      🎉 no goals
                                    -/


@[simp, norm_cast] lemma coe_neg (f : IncidenceAlgebra 𝕜 α) : ⇑(-f) = -f := rfl

@[simp, norm_cast] lemma coe_sub (f g : IncidenceAlgebra 𝕜 α) : ⇑(f - g) = f - g := rfl

lemma neg_apply (f : IncidenceAlgebra 𝕜 α) (a b : α) : (-f) a b = -f a b := rfl

lemma sub_apply (f g : IncidenceAlgebra 𝕜 α) (a b : α) : (f - g) a b = f a b - g a b := rfl


instance instAddGroup : AddGroup (IncidenceAlgebra 𝕜 α) :=
  DFunLike.coe_injective.addGroup _ coe_zero coe_add coe_neg coe_sub (fun _ _ ↦ rfl) fun _ _ ↦ rfl


instance instAddCommGroup [AddCommGroup 𝕜] [LE α] : AddCommGroup (IncidenceAlgebra 𝕜 α) :=
  DFunLike.coe_injective.addCommGroup _ coe_zero coe_add coe_neg coe_sub (fun _ _ ↦ rfl)
    fun _ _ ↦ rfl


/-- The unit incidence algebra is the delta function, whose entries are `0` except on the diagonal
where they are `1`. -/
instance instOne : One (IncidenceAlgebra 𝕜 α) :=
  ⟨⟨fun a b ↦ if a = b then 1 else 0, fun _a _b h ↦ ite_eq_right_iff.2 fun H ↦ (h H.le).elim⟩⟩


@[simp] lemma one_apply (a b : α) : (1 : IncidenceAlgebra 𝕜 α) a b = if a = b then 1 else 0 := rfl


/--
The multiplication operation in incidence algebras is defined on an interval by summing over
all divisions into two subintervals the product of the values of the original pair of functions.
-/
instance instMul : Mul (IncidenceAlgebra 𝕜 α) where
  mul f g :=
                                                            /-
                                                              F : Type u_1
                                                              𝕜 : Type u_2
                                                              𝕝 : Type u_3
                                                              𝕞 : Type u_4
                                                              α : Type u_5
                                                              β : Type u_6
                                                              inst✝³ : Preorder α
                                                              inst✝² : LocallyFiniteOrder α
                                                              inst✝¹ : AddCommMonoid 𝕜
                                                              inst✝ : Mul 𝕜
                                                              f g : IncidenceAlgebra 𝕜 α
                                                              a b : α
                                                              h : Not (LE.le a b)
                                                              ⊢ Eq ((fun a b => (Finset.Icc a b).sum fun x => HMul.hMul (f a x) (g x b)) a b …
                                                            -/
    ⟨fun a b ↦ ∑ x ∈ Icc a b, f a x * g x b, fun a b h ↦ by dsimp; rw [Icc_eq_empty h, sum_empty]⟩
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


@[simp] lemma mul_apply (f g : IncidenceAlgebra 𝕜 α) (a b : α) :
    (f * g) a b = ∑ x ∈ Icc a b, f a x * g x b := rfl


instance instNonUnitalNonAssocSemiring [Preorder α] [LocallyFiniteOrder α]
    [NonUnitalNonAssocSemiring 𝕜] : NonUnitalNonAssocSemiring (IncidenceAlgebra 𝕜 α) where
  __ := instAddCommMonoid
  mul := (· * ·)
  zero := 0
                         /-
                           F : Type u_1
                           𝕜 : Type u_2
                           𝕝 : Type u_3
                           𝕞 : Type u_4
                           α : Type u_5
                           β : Type u_6
                           inst✝² : Preorder α
                           inst✝¹ : LocallyFiniteOrder α
                           inst✝ : NonUnitalNonAssocSemiring 𝕜
                           f : IncidenceAlgebra 𝕜 α
                           ⊢ Eq (HMul.hMul 0 f) 0
                         -/
  zero_mul := fun f ↦ by ext; exact sum_eq_zero fun x _ ↦ MulZeroClass.zero_mul _
                              /-
                                🎉 no goals
                              -/
    /-
      F : Type u_1
      𝕜 : Type u_2
      𝕝 : Type u_3
      𝕞 : Type u_4
      α : Type u_5
      β : Type u_6
      inst✝² : Preorder α
      inst✝¹ : LocallyFiniteOrder α
      inst✝ : NonUnitalNonAssocSemiring 𝕜
      f g h : IncidenceAlgebra 𝕜 α
      ⊢ Eq (HMul.hMul f (HAdd.hAdd g h)) (HAdd.hAdd (HMul.hMul f g) (HMul.hMul f h))
    -/
                         /-
                           F : Type u_1
                           𝕜 : Type u_2
                           𝕝 : Type u_3
                           𝕞 : Type u_4
                           α : Type u_5
                           β : Type u_6
                           inst✝² : Preorder α
                           inst✝¹ : LocallyFiniteOrder α
                           inst✝ : NonUnitalNonAssocSemiring 𝕜
                           f : IncidenceAlgebra 𝕜 α
                           ⊢ Eq (HMul.hMul f 0) 0
                         -/
         /-
           🎉 no goals
         -/
  mul_zero := fun f ↦ by ext; exact sum_eq_zero fun x _ ↦ MulZeroClass.mul_zero _
    /-
      F : Type u_1
      𝕜 : Type u_2
      𝕝 : Type u_3
      𝕞 : Type u_4
      α : Type u_5
      β : Type u_6
      inst✝² : Preorder α
      inst✝¹ : LocallyFiniteOrder α
      inst✝ : NonUnitalNonAssocSemiring 𝕜
      f g h : IncidenceAlgebra 𝕜 α
      ⊢ Eq (HMul.hMul (HAdd.hAdd f g) h) (HAdd.hAdd (HMul.hMul f h) (HMul.hMul g h))
    -/
                              /-
                                🎉 no goals
                              -/
         /-
           🎉 no goals
         -/
  left_distrib := fun f g h ↦ by
    ext; exact Eq.trans (sum_congr rfl fun x _ ↦ left_distrib _ _ _) sum_add_distrib
  right_distrib := fun f g h ↦ by
    ext; exact Eq.trans (sum_congr rfl fun x _ ↦ right_distrib _ _ _) sum_add_distrib


instance instNonAssocSemiring [Preorder α] [LocallyFiniteOrder α] [DecidableEq α]
    [NonAssocSemiring 𝕜] : NonAssocSemiring (IncidenceAlgebra 𝕜 α) where
  __ := instNonUnitalNonAssocSemiring
  mul := (· * ·)
  zero := 0
  one := 1
                        /-
                          F : Type u_1
                          𝕜 : Type u_2
                          𝕝 : Type u_3
                          𝕞 : Type u_4
                          α : Type u_5
                          β : Type u_6
                          inst✝³ : Preorder α
                          inst✝² : LocallyFiniteOrder α
                          inst✝¹ : DecidableEq α
                          inst✝ : NonAssocSemiring 𝕜
                          f : IncidenceAlgebra 𝕜 α
                          ⊢ Eq (HMul.hMul 1 f) f
                        -/
  one_mul := fun f ↦ by ext; simp [*]
                             /-
                               🎉 no goals
                             -/
                        /-
                          F : Type u_1
                          𝕜 : Type u_2
                          𝕝 : Type u_3
                          𝕞 : Type u_4
                          α : Type u_5
                          β : Type u_6
                          inst✝³ : Preorder α
                          inst✝² : LocallyFiniteOrder α
                          inst✝¹ : DecidableEq α
                          inst✝ : NonAssocSemiring 𝕜
                          f : IncidenceAlgebra 𝕜 α
                          ⊢ Eq (HMul.hMul f 1) f
                        -/
  mul_one := fun f ↦ by ext; simp [*]
                             /-
                               🎉 no goals
                             -/


instance instSemiring [Preorder α] [LocallyFiniteOrder α] [DecidableEq α] [Semiring 𝕜] :
    Semiring (IncidenceAlgebra 𝕜 α) where
  __ := instNonAssocSemiring
  mul := (· * ·)
  mul_assoc f g h := by
    /-
      F : Type u_1
      𝕜 : Type u_2
      𝕝 : Type u_3
      𝕞 : Type u_4
      α : Type u_5
      β : Type u_6
      inst✝³ : Preorder α
      inst✝² : LocallyFiniteOrder α
      inst✝¹ : DecidableEq α
      inst✝ : Semiring 𝕜
      f g h : IncidenceAlgebra 𝕜 α
      ⊢ Eq (HMul.hMul (HMul.hMul f g) h) (HMul.hMul f (HMul.hMul g h))
    -/
    ext a b
    /-
      case h
      F : Type u_1
      𝕜 : Type u_2
      𝕝 : Type u_3
      𝕞 : Type u_4
      α : Type u_5
      β : Type u_6
      inst✝³ : Preorder α
      inst✝² : LocallyFiniteOrder α
      inst✝¹ : DecidableEq α
      inst✝ : Semiring 𝕜
      f g h : IncidenceAlgebra 𝕜 α
      a b : α
      a✝ : LE.le a b
      ⊢ Eq ((HMul.hMul (HMul.hMul f g) h) a b) ((HMul.hMul f (HMul.hMul g h)) a b)
    -/
    simp only [mul_apply, sum_mul, mul_sum, sum_sigma']
    /-
      case h
      F : Type u_1
      𝕜 : Type u_2
      𝕝 : Type u_3
      𝕞 : Type u_4
      α : Type u_5
      β : Type u_6
      inst✝³ : Preorder α
      inst✝² : LocallyFiniteOrder α
      inst✝¹ : DecidableEq α
      inst✝ : Semiring 𝕜
      f g h : IncidenceAlgebra 𝕜 α
      a b : α
      a✝ : LE.le a b
      ⊢ Eq (((Finset.Icc a b).sigma (Finset.Icc a)).sum fun x => HMul.hMul (HMul.hMu …
    -/
    apply sum_nbij' (fun ⟨a, b⟩ ↦ ⟨b, a⟩) (fun ⟨a, b⟩ ↦ ⟨b, a⟩) <;>
      /-
        case h.hi
        F : Type u_1
        𝕜 : Type u_2
        𝕝 : Type u_3
        𝕞 : Type u_4
        α : Type u_5
        β : Type u_6
        inst✝³ : Preorder α
        inst✝² : LocallyFiniteOrder α
        inst✝¹ : DecidableEq α
        inst✝ : Semiring 𝕜
        f g h : IncidenceAlgebra 𝕜 α
        a b : α
        a✝ : LE.le a b
        ⊢ ∀ (a_1 : Sigma fun i => α), Membership.mem ((Finset.Icc a b).sigma (Finset.I …
      -/
      /-
        🎉 no goals
      -/
      /-
        🎉 no goals
      -/
      /-
        🎉 no goals
      -/
      /-
        🎉 no goals
      -/
      aesop (add simp mul_assoc) (add unsafe le_trans)
      /-
        🎉 no goals
      -/
  one := 1
  zero := 0


instance instRing [Preorder α] [LocallyFiniteOrder α] [DecidableEq α] [Ring 𝕜] :
    Ring (IncidenceAlgebra 𝕜 α) where
  __ := instSemiring
  __ := instAddGroup


instance instSMul : SMul (IncidenceAlgebra 𝕜 α) (IncidenceAlgebra 𝕝 α) :=
  ⟨fun f g ↦
                                                            /-
                                                              F : Type u_1
                                                              𝕜 : Type u_2
                                                              𝕝 : Type u_3
                                                              𝕞 : Type u_4
                                                              α : Type u_5
                                                              β : Type u_6
                                                              inst✝⁴ : Preorder α
                                                              inst✝³ : LocallyFiniteOrder α
                                                              inst✝² : AddCommMonoid 𝕜
                                                              inst✝¹ : AddCommMonoid 𝕝
                                                              inst✝ : SMul 𝕜 𝕝
                                                              f : IncidenceAlgebra 𝕜 α
                                                              g : IncidenceAlgebra 𝕝 α
                                                              a b : α
                                                              h : Not (LE.le a b)
                                                              ⊢ Eq ((fun a b => (Finset.Icc a b).sum fun x => HSMul.hSMul (f a x) (g x b)) a …
                                                            -/
    ⟨fun a b ↦ ∑ x ∈ Icc a b, f a x • g x b, fun a b h ↦ by dsimp; rw [Icc_eq_empty h, sum_empty]⟩⟩
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


@[simp]
lemma smul_apply (f : IncidenceAlgebra 𝕜 α) (g : IncidenceAlgebra 𝕝 α) (a b : α) :
    (f • g) a b = ∑ x ∈ Icc a b, f a x • g x b :=
  rfl


instance instIsScalarTower [Preorder α] [LocallyFiniteOrder α] [AddCommMonoid 𝕜] [Monoid 𝕜]
    [Semiring 𝕝] [AddCommMonoid 𝕞] [SMul 𝕜 𝕝] [Module 𝕝 𝕞] [DistribMulAction 𝕜 𝕞]
    [IsScalarTower 𝕜 𝕝 𝕞] :
    IsScalarTower (IncidenceAlgebra 𝕜 α) (IncidenceAlgebra 𝕝 α) (IncidenceAlgebra 𝕞 α) where
  smul_assoc f g h := by
    /-
      F : Type u_1
      𝕜 : Type u_2
      𝕝 : Type u_3
      𝕞 : Type u_4
      α : Type u_5
      β : Type u_6
      inst✝⁹ : Preorder α
      inst✝⁸ : LocallyFiniteOrder α
      inst✝⁷ : AddCommMonoid 𝕜
      inst✝⁶ : Monoid 𝕜
      inst✝⁵ : Semiring 𝕝
      inst✝⁴ : AddCommMonoid 𝕞
      inst✝³ : SMul 𝕜 𝕝
      inst✝² : Module 𝕝 𝕞
      inst✝¹ : DistribMulAction 𝕜 𝕞
      inst✝ : IsScalarTower 𝕜 𝕝 𝕞
      f : IncidenceAlgebra 𝕜 α
      g : IncidenceAlgebra 𝕝 α
      h : IncidenceAlgebra 𝕞 α
      ⊢ Eq (HSMul.hSMul (HSMul.hSMul f g) h) (HSMul.hSMul f (HSMul.hSMul g h))
    -/
    ext a b
    /-
      case h
      F : Type u_1
      𝕜 : Type u_2
      𝕝 : Type u_3
      𝕞 : Type u_4
      α : Type u_5
      β : Type u_6
      inst✝⁹ : Preorder α
      inst✝⁸ : LocallyFiniteOrder α
      inst✝⁷ : AddCommMonoid 𝕜
      inst✝⁶ : Monoid 𝕜
      inst✝⁵ : Semiring 𝕝
      inst✝⁴ : AddCommMonoid 𝕞
      inst✝³ : SMul 𝕜 𝕝
      inst✝² : Module 𝕝 𝕞
      inst✝¹ : DistribMulAction 𝕜 𝕞
      inst✝ : IsScalarTower 𝕜 𝕝 𝕞
      f : IncidenceAlgebra 𝕜 α
      g : IncidenceAlgebra 𝕝 α
      h : IncidenceAlgebra 𝕞 α
      a b : α
      a✝ : LE.le a b
      ⊢ Eq ((HSMul.hSMul (HSMul.hSMul f g) h) a b) ((HSMul.hSMul f (HSMul.hSMul g h) …
    -/
    simp only [smul_apply, sum_smul, smul_sum, sum_sigma']
    /-
      case h
      F : Type u_1
      𝕜 : Type u_2
      𝕝 : Type u_3
      𝕞 : Type u_4
      α : Type u_5
      β : Type u_6
      inst✝⁹ : Preorder α
      inst✝⁸ : LocallyFiniteOrder α
      inst✝⁷ : AddCommMonoid 𝕜
      inst✝⁶ : Monoid 𝕜
      inst✝⁵ : Semiring 𝕝
      inst✝⁴ : AddCommMonoid 𝕞
      inst✝³ : SMul 𝕜 𝕝
      inst✝² : Module 𝕝 𝕞
      inst✝¹ : DistribMulAction 𝕜 𝕞
      inst✝ : IsScalarTower 𝕜 𝕝 𝕞
      f : IncidenceAlgebra 𝕜 α
      g : IncidenceAlgebra 𝕝 α
      h : IncidenceAlgebra 𝕞 α
      a b : α
      a✝ : LE.le a b
      ⊢ Eq (((Finset.Icc a b).sigma (Finset.Icc a)).sum fun x => HSMul.hSMul (HSMul. …
    -/
                                                                    /-
                                                                      🎉 no goals
                                                                    -/
                                                                    /-
                                                                      🎉 no goals
                                                                    -/
                                                                    /-
                                                                      🎉 no goals
                                                                    -/
                                                                    /-
                                                                      🎉 no goals
                                                                    -/
    apply sum_nbij' (fun ⟨a, b⟩ ↦ ⟨b, a⟩) (fun ⟨a, b⟩ ↦ ⟨b, a⟩) <;> aesop (add unsafe le_trans)
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


instance [Preorder α] [LocallyFiniteOrder α] [DecidableEq α] [Semiring 𝕜] [Semiring 𝕝]
    [Module 𝕜 𝕝] : Module (IncidenceAlgebra 𝕜 α) (IncidenceAlgebra 𝕝 α) where
  smul := (· • ·)
                   /-
                     F : Type u_1
                     𝕜 : Type u_2
                     𝕝 : Type u_3
                     𝕞 : Type u_4
                     α : Type u_5
                     β : Type u_6
                     inst✝⁵ : Preorder α
                     inst✝⁴ : LocallyFiniteOrder α
                     inst✝³ : DecidableEq α
                     inst✝² : Semiring 𝕜
                     inst✝¹ : Semiring 𝕝
                     inst✝ : Module 𝕜 𝕝
                     f : IncidenceAlgebra 𝕝 α
                     ⊢ Eq (HSMul.hSMul 1 f) f
                   -/
  one_smul f := by ext a b hab; simp [ite_smul, hab]
                                /-
                                  🎉 no goals
                                -/
  mul_smul := smul_assoc
                       /-
                         F : Type u_1
                         𝕜 : Type u_2
                         𝕝 : Type u_3
                         𝕞 : Type u_4
                         α : Type u_5
                         β : Type u_6
                         inst✝⁵ : Preorder α
                         inst✝⁴ : LocallyFiniteOrder α
                         inst✝³ : DecidableEq α
                         inst✝² : Semiring 𝕜
                         inst✝¹ : Semiring 𝕝
                         inst✝ : Module 𝕜 𝕝
                         f : IncidenceAlgebra 𝕜 α
                         g h : IncidenceAlgebra 𝕝 α
                         ⊢ Eq (HSMul.hSMul f (HAdd.hAdd g h)) (HAdd.hAdd (HSMul.hSMul f g) (HSMul.hSMul …
                       -/
  smul_add f g h := by ext; exact Eq.trans (sum_congr rfl fun x _ ↦ smul_add _ _ _) sum_add_distrib
                            /-
                              🎉 no goals
                            -/
                    /-
                      F : Type u_1
                      𝕜 : Type u_2
                      𝕝 : Type u_3
                      𝕞 : Type u_4
                      α : Type u_5
                      β : Type u_6
                      inst✝⁵ : Preorder α
                      inst✝⁴ : LocallyFiniteOrder α
                      inst✝³ : DecidableEq α
                      inst✝² : Semiring 𝕜
                      inst✝¹ : Semiring 𝕝
                      inst✝ : Module 𝕜 𝕝
                      f : IncidenceAlgebra 𝕜 α
                      ⊢ Eq (HSMul.hSMul f 0) 0
                    -/
                       /-
                         F : Type u_1
                         𝕜 : Type u_2
                         𝕝 : Type u_3
                         𝕞 : Type u_4
                         α : Type u_5
                         β : Type u_6
                         inst✝⁵ : Preorder α
                         inst✝⁴ : LocallyFiniteOrder α
                         inst✝³ : DecidableEq α
                         inst✝² : Semiring 𝕜
                         inst✝¹ : Semiring 𝕝
                         inst✝ : Module 𝕜 𝕝
                         f g : IncidenceAlgebra 𝕜 α
                         h : IncidenceAlgebra 𝕝 α
                         ⊢ Eq (HSMul.hSMul (HAdd.hAdd f g) h) (HAdd.hAdd (HSMul.hSMul f h) (HSMul.hSMul …
                       -/
                         /-
                           🎉 no goals
                         -/
  add_smul f g h := by ext; exact Eq.trans (sum_congr rfl fun x _ ↦ add_smul _ _ _) sum_add_distrib
                            /-
                              🎉 no goals
                            -/
                    /-
                      F : Type u_1
                      𝕜 : Type u_2
                      𝕝 : Type u_3
                      𝕞 : Type u_4
                      α : Type u_5
                      β : Type u_6
                      inst✝⁵ : Preorder α
                      inst✝⁴ : LocallyFiniteOrder α
                      inst✝³ : DecidableEq α
                      inst✝² : Semiring 𝕜
                      inst✝¹ : Semiring 𝕝
                      inst✝ : Module 𝕜 𝕝
                      f : IncidenceAlgebra 𝕝 α
                      ⊢ Eq (HSMul.hSMul 0 f) 0
                    -/
  zero_smul f := by ext; exact sum_eq_zero fun x _ ↦ zero_smul _ _
                         /-
                           🎉 no goals
                         -/
  smul_zero f := by ext; exact sum_eq_zero fun x _ ↦ smul_zero _


instance smulWithZeroRight [Zero 𝕜] [Zero 𝕝] [SMulWithZero 𝕜 𝕝] [LE α] :
    SMulWithZero 𝕜 (IncidenceAlgebra 𝕝 α) :=
  DFunLike.coe_injective.smulWithZero ⟨((⇑) : IncidenceAlgebra 𝕝 α → α → α → 𝕝), coe_zero⟩
    coe_constSMul


instance moduleRight [Preorder α] [Semiring 𝕜] [AddCommMonoid 𝕝] [Module 𝕜 𝕝] :
    Module 𝕜 (IncidenceAlgebra 𝕝 α) :=
  DFunLike.coe_injective.module _ ⟨⟨((⇑) : IncidenceAlgebra 𝕝 α → α → α → 𝕝), coe_zero⟩, coe_add⟩
    coe_constSMul


instance algebraRight [PartialOrder α] [LocallyFiniteOrder α] [DecidableEq α] [CommSemiring 𝕜]
    [CommSemiring 𝕝] [Algebra 𝕜 𝕝] : Algebra 𝕜 (IncidenceAlgebra 𝕝 α) where
  toFun c := algebraMap 𝕜 𝕝 c • (1 : IncidenceAlgebra 𝕝 α)
  map_one' := by
    /-
      F : Type u_1
      𝕜 : Type u_2
      𝕝 : Type u_3
      𝕞 : Type u_4
      α : Type u_5
      β : Type u_6
      inst✝⁵ : PartialOrder α
      inst✝⁴ : LocallyFiniteOrder α
      inst✝³ : DecidableEq α
      inst✝² : CommSemiring 𝕜
      inst✝¹ : CommSemiring 𝕝
      inst✝ : Algebra 𝕜 𝕝
      ⊢ Eq ((fun c => HSMul.hSMul ((algebraMap 𝕜 𝕝) c) 1) 1) 1
    -/
    ext; simp only [mul_boole, one_apply, Algebra.id.smul_eq_mul, constSMul_apply, map_one]
         /-
           🎉 no goals
         -/
  map_mul' c d := by
      /-
        F : Type u_1
        𝕜 : Type u_2
        𝕝 : Type u_3
        𝕞 : Type u_4
        α : Type u_5
        β : Type u_6
        inst✝⁵ : PartialOrder α
        inst✝⁴ : LocallyFiniteOrder α
        inst✝³ : DecidableEq α
        inst✝² : CommSemiring 𝕜
        inst✝¹ : CommSemiring 𝕝
        inst✝ : Algebra 𝕜 𝕝
        c d : 𝕜
        ⊢ Eq ({ toFun := fun c => HSMul.hSMul ((algebraMap 𝕜 𝕝) c) 1, map_one' := ⋯ }. …
      -/
      ext a b
      /-
        case h
        F : Type u_1
        𝕜 : Type u_2
        𝕝 : Type u_3
        𝕞 : Type u_4
        α : Type u_5
        β : Type u_6
        inst✝⁵ : PartialOrder α
        inst✝⁴ : LocallyFiniteOrder α
        inst✝³ : DecidableEq α
        inst✝² : CommSemiring 𝕜
        inst✝¹ : CommSemiring 𝕝
        inst✝ : Algebra 𝕜 𝕝
        c d : 𝕜
        a b : α
        a✝ : LE.le a b
        ⊢ Eq (({ toFun := fun c => HSMul.hSMul ((algebraMap 𝕜 𝕝) c) 1, map_one' := ⋯ } …
      -/
      obtain rfl | h := eq_or_ne a b
      · simp only [one_apply, Algebra.id.smul_eq_mul, mul_apply, Algebra.mul_smul_comm,
          boole_smul, constSMul_apply, ← ite_and, map_mul, Algebra.smul_mul_assoc,
          if_pos rfl, eq_comm, and_self_iff, Icc_self]
        /-
          case h.inl
          F : Type u_1
          𝕜 : Type u_2
          𝕝 : Type u_3
          𝕞 : Type u_4
          α : Type u_5
          β : Type u_6
          inst✝⁵ : PartialOrder α
          inst✝⁴ : LocallyFiniteOrder α
          inst✝³ : DecidableEq α
          inst✝² : CommSemiring 𝕜
          inst✝¹ : CommSemiring 𝕝
          inst✝ : Algebra 𝕜 𝕝
          c d : 𝕜
          a : α
          a✝ : LE.le a a
          ⊢ Eq (HMul.hMul (HMul.hMul ((algebraMap 𝕜 𝕝) c) ((algebraMap 𝕜 𝕝) d)) (ite Tru …
        -/
        simp
        /-
          🎉 no goals
        -/
      · simp only [true_and, ite_self, le_rfl, one_apply, mul_one, Algebra.id.smul_eq_mul,
          mul_apply, Algebra.mul_smul_comm, MulZeroClass.zero_mul, constSMul_apply,
          ← ite_and, ite_mul, mul_ite, map_mul, mem_Icc, sum_ite_eq,
          MulZeroClass.mul_zero, smul_zero, Algebra.smul_mul_assoc, if_pos rfl, if_neg h]
        /-
          case h.inr
          F : Type u_1
          𝕜 : Type u_2
          𝕝 : Type u_3
          𝕞 : Type u_4
          α : Type u_5
          β : Type u_6
          inst✝⁵ : PartialOrder α
          inst✝⁴ : LocallyFiniteOrder α
          inst✝³ : DecidableEq α
          inst✝² : CommSemiring 𝕜
          inst✝¹ : CommSemiring 𝕝
          inst✝ : Algebra 𝕜 𝕝
          c d : 𝕜
          a b : α
          a✝ : LE.le a b
          h : Ne a b
          ⊢ Eq 0 ((Finset.Icc a b).sum fun x => ite (And (Eq x b) (Eq a x)) (HMul.hMul ( …
        -/
        refine (sum_eq_zero fun x _ ↦ ?_).symm
        /-
          case h.inr
          F : Type u_1
          𝕜 : Type u_2
          𝕝 : Type u_3
          𝕞 : Type u_4
          α : Type u_5
          β : Type u_6
          inst✝⁵ : PartialOrder α
          inst✝⁴ : LocallyFiniteOrder α
          inst✝³ : DecidableEq α
          inst✝² : CommSemiring 𝕜
          inst✝¹ : CommSemiring 𝕝
          inst✝ : Algebra 𝕜 𝕝
          c d : 𝕜
          a b : α
          a✝ : LE.le a b
          h : Ne a b
          x : α
          x✝ : Membership.mem (Finset.Icc a b) x
          ⊢ Eq (ite (And (Eq x b) (Eq a x)) (HMul.hMul ((algebraMap 𝕜 𝕝) c) ((algebraMap …
        -/
        exact if_neg fun hx ↦ h <| hx.2.trans hx.1
        /-
          🎉 no goals
        -/
                  /-
                    F : Type u_1
                    𝕜 : Type u_2
                    𝕝 : Type u_3
                    𝕞 : Type u_4
                    α : Type u_5
                    β : Type u_6
                    inst✝⁵ : PartialOrder α
                    inst✝⁴ : LocallyFiniteOrder α
                    inst✝³ : DecidableEq α
                    inst✝² : CommSemiring 𝕜
                    inst✝¹ : CommSemiring 𝕝
                    inst✝ : Algebra 𝕜 𝕝
                    ⊢ Eq ((↑{ toFun := fun c => HSMul.hSMul ((algebraMap 𝕜 𝕝) c) 1, map_one' := ⋯, …
                  -/
  map_zero' := by dsimp; rw [map_zero, zero_smul]
                         /-
                           🎉 no goals
                         -/
                     /-
                       F : Type u_1
                       𝕜 : Type u_2
                       𝕝 : Type u_3
                       𝕞 : Type u_4
                       α : Type u_5
                       β : Type u_6
                       inst✝⁵ : PartialOrder α
                       inst✝⁴ : LocallyFiniteOrder α
                       inst✝³ : DecidableEq α
                       inst✝² : CommSemiring 𝕜
                       inst✝¹ : CommSemiring 𝕝
                       inst✝ : Algebra 𝕜 𝕝
                       c d : 𝕜
                       ⊢ Eq ((↑{ toFun := fun c => HSMul.hSMul ((algebraMap 𝕜 𝕝) c) 1, map_one' := ⋯, …
                     -/
  map_add' c d := by dsimp; rw [map_add, add_smul]
                            /-
                              🎉 no goals
                            -/
                      /-
                        F : Type u_1
                        𝕜 : Type u_2
                        𝕝 : Type u_3
                        𝕞 : Type u_4
                        α : Type u_5
                        β : Type u_6
                        inst✝⁵ : PartialOrder α
                        inst✝⁴ : LocallyFiniteOrder α
                        inst✝³ : DecidableEq α
                        inst✝² : CommSemiring 𝕜
                        inst✝¹ : CommSemiring 𝕝
                        inst✝ : Algebra 𝕜 𝕝
                        c : 𝕜
                        f : IncidenceAlgebra 𝕝 α
                        ⊢ Eq (HMul.hMul ({ toFun := fun c => HSMul.hSMul ((algebraMap 𝕜 𝕝) c) 1, map_o …
                      -/
  commutes' c f := by classical ext a b hab; simp [if_pos hab, constSMul_apply, mul_comm]
                      /-
                        🎉 no goals
                      -/
                      /-
                        F : Type u_1
                        𝕜 : Type u_2
                        𝕝 : Type u_3
                        𝕞 : Type u_4
                        α : Type u_5
                        β : Type u_6
                        inst✝⁵ : PartialOrder α
                        inst✝⁴ : LocallyFiniteOrder α
                        inst✝³ : DecidableEq α
                        inst✝² : CommSemiring 𝕜
                        inst✝¹ : CommSemiring 𝕝
                        inst✝ : Algebra 𝕜 𝕝
                        c : 𝕜
                        f : IncidenceAlgebra 𝕝 α
                        ⊢ Eq (HSMul.hSMul c f) (HMul.hMul ({ toFun := fun c => HSMul.hSMul ((algebraMa …
                      -/
  smul_def' c f := by classical ext a b hab; simp [if_pos hab, constSMul_apply, Algebra.smul_def]
                      /-
                        🎉 no goals
                      -/


/-- The lambda function of the incidence algebra is the function that assigns `1` to every nonempty
interval of cardinality one or two. -/
@[simps]
def lambda : IncidenceAlgebra 𝕜 α :=
  ⟨fun a b ↦ if a ⩿ b then 1 else 0, fun _a _b h ↦ if_neg fun hh ↦ h hh.le⟩


/-- The zeta function of the incidence algebra is the function that assigns 1 to every nonempty
interval, convolution with this function sums functions over intervals. -/
def zeta : IncidenceAlgebra 𝕜 α := ⟨fun a b ↦ if a ≤ b then 1 else 0, fun _a _b h ↦ if_neg h⟩


@[simp] lemma zeta_apply (a b : α) : zeta 𝕜 a b = if a ≤ b then 1 else 0 := rfl


lemma zeta_of_le (h : a ≤ b) : zeta 𝕜 a b = 1 := if_pos h


lemma zeta_mul_zeta [Semiring 𝕜] [Preorder α] [LocallyFiniteOrder α] [DecidableRel (α := α) (· ≤ ·)]
    (a b : α) : (zeta 𝕜 * zeta 𝕜 : IncidenceAlgebra 𝕜 α) a b = (Icc a b).card := by
  /-
    𝕜 : Type u_2
    α : Type u_5
    inst✝³ : Semiring 𝕜
    inst✝² : Preorder α
    inst✝¹ : LocallyFiniteOrder α
    inst✝ : DecidableRel fun x1 x2 => LE.le x1 x2
    a b : α
    ⊢ Eq ((HMul.hMul (IncidenceAlgebra.zeta 𝕜) (IncidenceAlgebra.zeta 𝕜)) a b) ↑(F …
  -/
  rw [mul_apply, card_eq_sum_ones, Nat.cast_sum, Nat.cast_one]
  /-
    𝕜 : Type u_2
    α : Type u_5
    inst✝³ : Semiring 𝕜
    inst✝² : Preorder α
    inst✝¹ : LocallyFiniteOrder α
    inst✝ : DecidableRel fun x1 x2 => LE.le x1 x2
    a b : α
    ⊢ Eq ((Finset.Icc a b).sum fun x => HMul.hMul ((IncidenceAlgebra.zeta 𝕜) a x)  …
  -/
  refine sum_congr rfl fun x hx ↦ ?_
  /-
    𝕜 : Type u_2
    α : Type u_5
    inst✝³ : Semiring 𝕜
    inst✝² : Preorder α
    inst✝¹ : LocallyFiniteOrder α
    inst✝ : DecidableRel fun x1 x2 => LE.le x1 x2
    a b x : α
    hx : Membership.mem (Finset.Icc a b) x
    ⊢ Eq (HMul.hMul ((IncidenceAlgebra.zeta 𝕜) a x) ((IncidenceAlgebra.zeta 𝕜) x b …
  -/
  rw [mem_Icc] at hx
  /-
    𝕜 : Type u_2
    α : Type u_5
    inst✝³ : Semiring 𝕜
    inst✝² : Preorder α
    inst✝¹ : LocallyFiniteOrder α
    inst✝ : DecidableRel fun x1 x2 => LE.le x1 x2
    a b x : α
    hx : And (LE.le a x) (LE.le x b)
    ⊢ Eq (HMul.hMul ((IncidenceAlgebra.zeta 𝕜) a x) ((IncidenceAlgebra.zeta 𝕜) x b …
  -/
  rw [zeta_of_le hx.1, zeta_of_le hx.2, one_mul]
  /-
    🎉 no goals
  -/


lemma zeta_mul_kappa [Semiring 𝕜] [Preorder α] [LocallyFiniteOrder α]
    [DecidableRel (α := α) (· ≤ ·)] (a b : α) :
    (zeta 𝕜 * zeta 𝕜 : IncidenceAlgebra 𝕜 α) a b = (Icc a b).card := by
  /-
    𝕜 : Type u_2
    α : Type u_5
    inst✝³ : Semiring 𝕜
    inst✝² : Preorder α
    inst✝¹ : LocallyFiniteOrder α
    inst✝ : DecidableRel fun x1 x2 => LE.le x1 x2
    a b : α
    ⊢ Eq ((HMul.hMul (IncidenceAlgebra.zeta 𝕜) (IncidenceAlgebra.zeta 𝕜)) a b) ↑(F …
  -/
  rw [mul_apply, card_eq_sum_ones, Nat.cast_sum, Nat.cast_one]
  /-
    𝕜 : Type u_2
    α : Type u_5
    inst✝³ : Semiring 𝕜
    inst✝² : Preorder α
    inst✝¹ : LocallyFiniteOrder α
    inst✝ : DecidableRel fun x1 x2 => LE.le x1 x2
    a b : α
    ⊢ Eq ((Finset.Icc a b).sum fun x => HMul.hMul ((IncidenceAlgebra.zeta 𝕜) a x)  …
  -/
  refine sum_congr rfl fun x hx ↦ ?_
  /-
    𝕜 : Type u_2
    α : Type u_5
    inst✝³ : Semiring 𝕜
    inst✝² : Preorder α
    inst✝¹ : LocallyFiniteOrder α
    inst✝ : DecidableRel fun x1 x2 => LE.le x1 x2
    a b x : α
    hx : Membership.mem (Finset.Icc a b) x
    ⊢ Eq (HMul.hMul ((IncidenceAlgebra.zeta 𝕜) a x) ((IncidenceAlgebra.zeta 𝕜) x b …
  -/
  rw [mem_Icc] at hx
  /-
    𝕜 : Type u_2
    α : Type u_5
    inst✝³ : Semiring 𝕜
    inst✝² : Preorder α
    inst✝¹ : LocallyFiniteOrder α
    inst✝ : DecidableRel fun x1 x2 => LE.le x1 x2
    a b x : α
    hx : And (LE.le a x) (LE.le x b)
    ⊢ Eq (HMul.hMul ((IncidenceAlgebra.zeta 𝕜) a x) ((IncidenceAlgebra.zeta 𝕜) x b …
  -/
  rw [zeta_of_le hx.1, zeta_of_le hx.2, one_mul]
  /-
    🎉 no goals
  -/


/-- The Möbius function of the incidence algebra as a bare function defined recursively. -/
private def muFun (a : α) : α → 𝕜
  | b =>
    if a = b then 1
    else
      -∑ x in (Ico a b).attach,
          let h := mem_Ico.1 x.2
          have : (Icc a x).card < (Icc a b).card :=
            card_lt_card (Icc_ssubset_Icc_right (h.1.trans h.2.le) le_rfl h.2)
          muFun a x
termination_by b => (Icc a b).card


private lemma muFun_apply (a b : α) :
                                                                                   /-
                                                                                     𝕜 : Type u_2
                                                                                     α : Type u_5
                                                                                     inst✝⁴ : AddCommGroup 𝕜
                                                                                     inst✝³ : One 𝕜
                                                                                     inst✝² : Preorder α
                                                                                     inst✝¹ : LocallyFiniteOrder α
                                                                                     inst✝ : DecidableEq α
                                                                                     a b : α
                                                                                     ⊢ Eq (IncidenceAlgebra.muFun 𝕜 a b) (ite (Eq a b) 1 (Neg.neg ((Finset.Ico a b) …
                                                                                   -/
    muFun 𝕜 a b = if a = b then 1 else -∑ x in (Ico a b).attach, muFun 𝕜 a x := by rw [muFun]
                                                                                   /-
                                                                                     🎉 no goals
                                                                                   -/


/-- The Möbius function which inverts `zeta` as an element of the incidence algebra. -/
def mu : IncidenceAlgebra 𝕜 α :=
  ⟨muFun 𝕜, fun a b ↦ not_imp_comm.1 fun h ↦ by
    /-
      F : Type u_1
      𝕜 : Type u_2
      𝕝 : Type u_3
      𝕞 : Type u_4
      α : Type u_5
      β : Type u_6
      inst✝⁴ : AddCommGroup 𝕜
      inst✝³ : One 𝕜
      inst✝² : Preorder α
      inst✝¹ : LocallyFiniteOrder α
      inst✝ : DecidableEq α
      a b : α
      h : Not (Eq (IncidenceAlgebra.muFun 𝕜 a b) 0)
      ⊢ LE.le a b
    -/
    rw [muFun_apply] at h
    /-
      F : Type u_1
      𝕜 : Type u_2
      𝕝 : Type u_3
      𝕞 : Type u_4
      α : Type u_5
      β : Type u_6
      inst✝⁴ : AddCommGroup 𝕜
      inst✝³ : One 𝕜
      inst✝² : Preorder α
      inst✝¹ : LocallyFiniteOrder α
      inst✝ : DecidableEq α
      a b : α
      h : Not (Eq (ite (Eq a b) 1 (Neg.neg ((Finset.Ico a b).attach.sum fun x => Inc …
      ⊢ LE.le a b
    -/
    split_ifs at h  with hab
      /-
        case pos
        F : Type u_1
        𝕜 : Type u_2
        𝕝 : Type u_3
        𝕞 : Type u_4
        α : Type u_5
        β : Type u_6
        inst✝⁴ : AddCommGroup 𝕜
        inst✝³ : One 𝕜
        inst✝² : Preorder α
        inst✝¹ : LocallyFiniteOrder α
        inst✝ : DecidableEq α
        a b : α
        hab : Eq a b
        h : Not (Eq 1 0)
        ⊢ LE.le a b
      -/
    · exact hab.le
      /-
        🎉 no goals
      -/
      /-
        case neg
        F : Type u_1
        𝕜 : Type u_2
        𝕝 : Type u_3
        𝕞 : Type u_4
        α : Type u_5
        β : Type u_6
        inst✝⁴ : AddCommGroup 𝕜
        inst✝³ : One 𝕜
        inst✝² : Preorder α
        inst✝¹ : LocallyFiniteOrder α
        inst✝ : DecidableEq α
        a b : α
        hab : Not (Eq a b)
        h : Not (Eq (Neg.neg ((Finset.Ico a b).attach.sum fun x => IncidenceAlgebra.mu …
        ⊢ LE.le a b
      -/
    · rw [neg_eq_zero] at h
      /-
        case neg
        F : Type u_1
        𝕜 : Type u_2
        𝕝 : Type u_3
        𝕞 : Type u_4
        α : Type u_5
        β : Type u_6
        inst✝⁴ : AddCommGroup 𝕜
        inst✝³ : One 𝕜
        inst✝² : Preorder α
        inst✝¹ : LocallyFiniteOrder α
        inst✝ : DecidableEq α
        a b : α
        hab : Not (Eq a b)
        h : Not (Eq ((Finset.Ico a b).attach.sum fun x => IncidenceAlgebra.muFun 𝕜 a ↑ …
        ⊢ LE.le a b
      -/
      obtain ⟨⟨x, hx⟩, -⟩ := exists_ne_zero_of_sum_ne_zero h
      /-
        case neg.intro.mk
        F : Type u_1
        𝕜 : Type u_2
        𝕝 : Type u_3
        𝕞 : Type u_4
        α : Type u_5
        β : Type u_6
        inst✝⁴ : AddCommGroup 𝕜
        inst✝³ : One 𝕜
        inst✝² : Preorder α
        inst✝¹ : LocallyFiniteOrder α
        inst✝ : DecidableEq α
        a b : α
        hab : Not (Eq a b)
        h : Not (Eq ((Finset.Ico a b).attach.sum fun x => IncidenceAlgebra.muFun 𝕜 a ↑ …
        x : α
        hx : Membership.mem (Finset.Ico a b) x
        ⊢ LE.le a b
      -/
      exact (nonempty_Ico.1 ⟨x, hx⟩).le⟩
      /-
        🎉 no goals
      -/


lemma mu_apply (a b : α) : mu 𝕜 a b = if a = b then 1 else -∑ x ∈ Ico a b, mu 𝕜 a x := by
  /-
    𝕜 : Type u_2
    α : Type u_5
    inst✝⁴ : AddCommGroup 𝕜
    inst✝³ : One 𝕜
    inst✝² : Preorder α
    inst✝¹ : LocallyFiniteOrder α
    inst✝ : DecidableEq α
    a b : α
    ⊢ Eq ((IncidenceAlgebra.mu 𝕜) a b) (ite (Eq a b) 1 (Neg.neg ((Finset.Ico a b). …
  -/
  rw [mu, coe_mk, muFun_apply, sum_attach]
  /-
    🎉 no goals
  -/


                                                   /-
                                                     𝕜 : Type u_2
                                                     α : Type u_5
                                                     inst✝⁴ : AddCommGroup 𝕜
                                                     inst✝³ : One 𝕜
                                                     inst✝² : Preorder α
                                                     inst✝¹ : LocallyFiniteOrder α
                                                     inst✝ : DecidableEq α
                                                     a : α
                                                     ⊢ Eq ((IncidenceAlgebra.mu 𝕜) a a) 1
                                                   -/
@[simp] lemma mu_self (a : α) : mu 𝕜 a a = 1 := by simp [mu_apply]
                                                   /-
                                                     🎉 no goals
                                                   -/


lemma mu_eq_neg_sum_Ico_of_ne (hab : a ≠ b) :
                                              /-
                                                𝕜 : Type u_2
                                                α : Type u_5
                                                inst✝⁴ : AddCommGroup 𝕜
                                                inst✝³ : One 𝕜
                                                inst✝² : Preorder α
                                                inst✝¹ : LocallyFiniteOrder α
                                                inst✝ : DecidableEq α
                                                a b : α
                                                hab : Ne a b
                                                ⊢ Eq ((IncidenceAlgebra.mu 𝕜) a b) (Neg.neg ((Finset.Ico a b).sum fun x => (In …
                                              -/
    mu 𝕜 a b = -∑ x ∈ Ico a b, mu 𝕜 a x := by rw [mu_apply, if_neg hab]
                                              /-
                                                🎉 no goals
                                              -/


lemma sum_Icc_mu_right (a b : α) : ∑ x ∈ Icc a b, mu 𝕜 a x = if a = b then 1 else 0 := by
  /-
    𝕜 : Type u_2
    α : Type u_5
    inst✝⁴ : AddCommGroup 𝕜
    inst✝³ : One 𝕜
    inst✝² : PartialOrder α
    inst✝¹ : LocallyFiniteOrder α
    inst✝ : DecidableEq α
    a b : α
    ⊢ Eq ((Finset.Icc a b).sum fun x => (IncidenceAlgebra.mu 𝕜) a x) (ite (Eq a b) …
  -/
  split_ifs with hab
    /-
      case pos
      𝕜 : Type u_2
      α : Type u_5
      inst✝⁴ : AddCommGroup 𝕜
      inst✝³ : One 𝕜
      inst✝² : PartialOrder α
      inst✝¹ : LocallyFiniteOrder α
      inst✝ : DecidableEq α
      a b : α
      hab : Eq a b
      ⊢ Eq ((Finset.Icc a b).sum fun x => (IncidenceAlgebra.mu 𝕜) a x) 1
    -/
  · simp [hab]
    /-
      🎉 no goals
    -/
  /-
    case neg
    𝕜 : Type u_2
    α : Type u_5
    inst✝⁴ : AddCommGroup 𝕜
    inst✝³ : One 𝕜
    inst✝² : PartialOrder α
    inst✝¹ : LocallyFiniteOrder α
    inst✝ : DecidableEq α
    a b : α
    hab : Not (Eq a b)
    ⊢ Eq ((Finset.Icc a b).sum fun x => (IncidenceAlgebra.mu 𝕜) a x) 0
  -/
  by_cases hab : a ≤ b
    /-
      case pos
      𝕜 : Type u_2
      α : Type u_5
      inst✝⁴ : AddCommGroup 𝕜
      inst✝³ : One 𝕜
      inst✝² : PartialOrder α
      inst✝¹ : LocallyFiniteOrder α
      inst✝ : DecidableEq α
      a b : α
      hab✝ : Not (Eq a b)
      hab : LE.le a b
      ⊢ Eq ((Finset.Icc a b).sum fun x => (IncidenceAlgebra.mu 𝕜) a x) 0
    -/
  · simp [Icc_eq_cons_Ico hab, mu_eq_neg_sum_Ico_of_ne ‹_›]
    /-
      🎉 no goals
    -/
  · exact sum_eq_zero fun x hx ↦ apply_eq_zero_of_not_le
      (fun hax ↦ hab <| hax.trans (mem_Icc.1 hx).2) _


/-- `mu'` as a bare function defined recursively. -/
private def muFun' (b : α) : α → 𝕜
  | a =>
    if a = b then 1
    else
      -∑ x in (Ioc a b).attach,
          let h := mem_Ioc.1 x.2
          have : (Icc ↑x b).card < (Icc a b).card :=
            card_lt_card (Icc_ssubset_Icc_left (h.1.le.trans h.2) h.1 le_rfl)
          muFun' b x
termination_by a => (Icc a b).card


private lemma muFun'_apply (a b : α) :
    muFun' 𝕜 b a = if a = b then 1 else -∑ x in (Ioc a b).attach, muFun' 𝕜 b x := by
  /-
    𝕜 : Type u_2
    α : Type u_5
    inst✝⁴ : AddCommGroup 𝕜
    inst✝³ : One 𝕜
    inst✝² : Preorder α
    inst✝¹ : LocallyFiniteOrder α
    inst✝ : DecidableEq α
    a b : α
    ⊢ Eq (IncidenceAlgebra.muFun' 𝕜 b a) (ite (Eq a b) 1 (Neg.neg ((Finset.Ioc a b …
  -/
  rw [muFun']
  /-
    🎉 no goals
  -/


/-- This is the reversed definition of `mu`, which is equal to `mu` but easiest to prove equal by
showing that `zeta * mu = 1` and `mu' * zeta = 1`. -/
private def mu' : IncidenceAlgebra 𝕜 α :=
  ⟨fun a b ↦ muFun' 𝕜 b a, fun a b ↦
    not_imp_comm.1 fun h ↦ by
      /-
        F : Type u_1
        𝕜 : Type u_2
        𝕝 : Type u_3
        𝕞 : Type u_4
        α : Type u_5
        β : Type u_6
        inst✝⁴ : AddCommGroup 𝕜
        inst✝³ : One 𝕜
        inst✝² : Preorder α
        inst✝¹ : LocallyFiniteOrder α
        inst✝ : DecidableEq α
        a b : α
        h : Not (Eq ((fun a b => IncidenceAlgebra.muFun' 𝕜 b a) a b) 0)
        ⊢ LE.le a b
      -/
      dsimp only at h
      /-
        F : Type u_1
        𝕜 : Type u_2
        𝕝 : Type u_3
        𝕞 : Type u_4
        α : Type u_5
        β : Type u_6
        inst✝⁴ : AddCommGroup 𝕜
        inst✝³ : One 𝕜
        inst✝² : Preorder α
        inst✝¹ : LocallyFiniteOrder α
        inst✝ : DecidableEq α
        a b : α
        h : Not (Eq (IncidenceAlgebra.muFun' 𝕜 b a) 0)
        ⊢ LE.le a b
      -/
      rw [muFun'_apply] at h
      /-
        F : Type u_1
        𝕜 : Type u_2
        𝕝 : Type u_3
        𝕞 : Type u_4
        α : Type u_5
        β : Type u_6
        inst✝⁴ : AddCommGroup 𝕜
        inst✝³ : One 𝕜
        inst✝² : Preorder α
        inst✝¹ : LocallyFiniteOrder α
        inst✝ : DecidableEq α
        a b : α
        h : Not (Eq (ite (Eq a b) 1 (Neg.neg ((Finset.Ioc a b).attach.sum fun x => Inc …
        ⊢ LE.le a b
      -/
      split_ifs at h  with hab
        /-
          case pos
          F : Type u_1
          𝕜 : Type u_2
          𝕝 : Type u_3
          𝕞 : Type u_4
          α : Type u_5
          β : Type u_6
          inst✝⁴ : AddCommGroup 𝕜
          inst✝³ : One 𝕜
          inst✝² : Preorder α
          inst✝¹ : LocallyFiniteOrder α
          inst✝ : DecidableEq α
          a b : α
          hab : Eq a b
          h : Not (Eq 1 0)
          ⊢ LE.le a b
        -/
      · exact hab.le
        /-
          🎉 no goals
        -/
        /-
          case neg
          F : Type u_1
          𝕜 : Type u_2
          𝕝 : Type u_3
          𝕞 : Type u_4
          α : Type u_5
          β : Type u_6
          inst✝⁴ : AddCommGroup 𝕜
          inst✝³ : One 𝕜
          inst✝² : Preorder α
          inst✝¹ : LocallyFiniteOrder α
          inst✝ : DecidableEq α
          a b : α
          hab : Not (Eq a b)
          h : Not (Eq (Neg.neg ((Finset.Ioc a b).attach.sum fun x => IncidenceAlgebra.mu …
          ⊢ LE.le a b
        -/
      · rw [neg_eq_zero] at h
        /-
          case neg
          F : Type u_1
          𝕜 : Type u_2
          𝕝 : Type u_3
          𝕞 : Type u_4
          α : Type u_5
          β : Type u_6
          inst✝⁴ : AddCommGroup 𝕜
          inst✝³ : One 𝕜
          inst✝² : Preorder α
          inst✝¹ : LocallyFiniteOrder α
          inst✝ : DecidableEq α
          a b : α
          hab : Not (Eq a b)
          h : Not (Eq ((Finset.Ioc a b).attach.sum fun x => IncidenceAlgebra.muFun' 𝕜 b  …
          ⊢ LE.le a b
        -/
        obtain ⟨⟨x, hx⟩, -⟩ := exists_ne_zero_of_sum_ne_zero h
        /-
          case neg.intro.mk
          F : Type u_1
          𝕜 : Type u_2
          𝕝 : Type u_3
          𝕞 : Type u_4
          α : Type u_5
          β : Type u_6
          inst✝⁴ : AddCommGroup 𝕜
          inst✝³ : One 𝕜
          inst✝² : Preorder α
          inst✝¹ : LocallyFiniteOrder α
          inst✝ : DecidableEq α
          a b : α
          hab : Not (Eq a b)
          h : Not (Eq ((Finset.Ioc a b).attach.sum fun x => IncidenceAlgebra.muFun' 𝕜 b  …
          x : α
          hx : Membership.mem (Finset.Ioc a b) x
          ⊢ LE.le a b
        -/
        exact (nonempty_Ioc.1 ⟨x, hx⟩).le⟩
        /-
          🎉 no goals
        -/


private lemma mu'_apply (a b : α) : mu' 𝕜 a b = if a = b then 1 else -∑ x ∈ Ioc a b, mu' 𝕜 x b := by
  /-
    𝕜 : Type u_2
    α : Type u_5
    inst✝⁴ : AddCommGroup 𝕜
    inst✝³ : One 𝕜
    inst✝² : Preorder α
    inst✝¹ : LocallyFiniteOrder α
    inst✝ : DecidableEq α
    a b : α
    ⊢ Eq ((IncidenceAlgebra.mu' 𝕜) a b) (ite (Eq a b) 1 (Neg.neg ((Finset.Ioc a b) …
  -/
  rw [mu', coe_mk, muFun'_apply, sum_attach]
  /-
    🎉 no goals
  -/


                                                                   /-
                                                                     𝕜 : Type u_2
                                                                     α : Type u_5
                                                                     inst✝⁴ : AddCommGroup 𝕜
                                                                     inst✝³ : One 𝕜
                                                                     inst✝² : Preorder α
                                                                     inst✝¹ : LocallyFiniteOrder α
                                                                     inst✝ : DecidableEq α
                                                                     a : α
                                                                     ⊢ Eq ((IncidenceAlgebra.mu' 𝕜) a a) 1
                                                                   -/
@[simp] private lemma mu'_apply_self (a : α) : mu' 𝕜 a a = 1 := by simp [mu'_apply]
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


private lemma mu'_eq_sum_Ioc_of_ne (h : a ≠ b) : mu' 𝕜 a b = -∑ x ∈ Ioc a b, mu' 𝕜 x b := by
  /-
    𝕜 : Type u_2
    α : Type u_5
    inst✝⁴ : AddCommGroup 𝕜
    inst✝³ : One 𝕜
    inst✝² : Preorder α
    inst✝¹ : LocallyFiniteOrder α
    inst✝ : DecidableEq α
    a b : α
    h : Ne a b
    ⊢ Eq ((IncidenceAlgebra.mu' 𝕜) a b) (Neg.neg ((Finset.Ioc a b).sum fun x => (I …
  -/
  rw [mu'_apply, if_neg h]
  /-
    🎉 no goals
  -/


private lemma sum_Icc_mu'_left (a b : α) : ∑ x ∈ Icc a b, mu' 𝕜 x b = if a = b then 1 else 0 := by
  /-
    𝕜 : Type u_2
    α : Type u_5
    inst✝⁴ : AddCommGroup 𝕜
    inst✝³ : One 𝕜
    inst✝² : PartialOrder α
    inst✝¹ : LocallyFiniteOrder α
    inst✝ : DecidableEq α
    a b : α
    ⊢ Eq ((Finset.Icc a b).sum fun x => (IncidenceAlgebra.mu' 𝕜) x b) (ite (Eq a b …
  -/
  split_ifs with hab
    /-
      case pos
      𝕜 : Type u_2
      α : Type u_5
      inst✝⁴ : AddCommGroup 𝕜
      inst✝³ : One 𝕜
      inst✝² : PartialOrder α
      inst✝¹ : LocallyFiniteOrder α
      inst✝ : DecidableEq α
      a b : α
      hab : Eq a b
      ⊢ Eq ((Finset.Icc a b).sum fun x => (IncidenceAlgebra.mu' 𝕜) x b) 1
    -/
  · simp [hab]
    /-
      🎉 no goals
    -/
  /-
    case neg
    𝕜 : Type u_2
    α : Type u_5
    inst✝⁴ : AddCommGroup 𝕜
    inst✝³ : One 𝕜
    inst✝² : PartialOrder α
    inst✝¹ : LocallyFiniteOrder α
    inst✝ : DecidableEq α
    a b : α
    hab : Not (Eq a b)
    ⊢ Eq ((Finset.Icc a b).sum fun x => (IncidenceAlgebra.mu' 𝕜) x b) 0
  -/
  by_cases hab : a ≤ b
    /-
      case pos
      𝕜 : Type u_2
      α : Type u_5
      inst✝⁴ : AddCommGroup 𝕜
      inst✝³ : One 𝕜
      inst✝² : PartialOrder α
      inst✝¹ : LocallyFiniteOrder α
      inst✝ : DecidableEq α
      a b : α
      hab✝ : Not (Eq a b)
      hab : LE.le a b
      ⊢ Eq ((Finset.Icc a b).sum fun x => (IncidenceAlgebra.mu' 𝕜) x b) 0
    -/
  · simp [Icc_eq_cons_Ioc hab, mu'_eq_sum_Ioc_of_ne ‹_›]
    /-
      🎉 no goals
    -/
  · exact sum_eq_zero fun x hx ↦ apply_eq_zero_of_not_le
      (fun hxb ↦ hab <| (mem_Icc.1 hx).1.trans hxb) _


lemma mu_mul_zeta : (mu 𝕜 * zeta 𝕜 : IncidenceAlgebra 𝕜 α) = 1 := by
  /-
    𝕜 : Type u_2
    α : Type u_5
    inst✝⁵ : AddCommGroup 𝕜
    inst✝⁴ : MulOneClass 𝕜
    inst✝³ : PartialOrder α
    inst✝² : LocallyFiniteOrder α
    inst✝¹ : DecidableEq α
    inst✝ : DecidableRel fun x1 x2 => LE.le x1 x2
    ⊢ Eq (HMul.hMul (IncidenceAlgebra.mu 𝕜) (IncidenceAlgebra.zeta 𝕜)) 1
  -/
  ext a b
  calc
    _ = ∑ x ∈ Icc a b, mu 𝕜 a x := by rw [mul_apply]; congr! with x hx; simp [(mem_Icc.1 hx).2]
    _ = (1 : IncidenceAlgebra 𝕜 α) a b := sum_Icc_mu_right ..


private lemma zeta_mul_mu' : (zeta 𝕜 * mu' 𝕜 : IncidenceAlgebra 𝕜 α) = 1 := by
  /-
    𝕜 : Type u_2
    α : Type u_5
    inst✝⁵ : AddCommGroup 𝕜
    inst✝⁴ : MulOneClass 𝕜
    inst✝³ : PartialOrder α
    inst✝² : LocallyFiniteOrder α
    inst✝¹ : DecidableEq α
    inst✝ : DecidableRel fun x1 x2 => LE.le x1 x2
    ⊢ Eq (HMul.hMul (IncidenceAlgebra.zeta 𝕜) (IncidenceAlgebra.mu' 𝕜)) 1
  -/
  ext a b
  calc
    _ = ∑ x ∈ Icc a b, mu' 𝕜 x b := by rw [mul_apply]; congr! with x hx; simp [(mem_Icc.1 hx).1]
    _ = (1 : IncidenceAlgebra 𝕜 α) a b := sum_Icc_mu'_left ..


private lemma mu_eq_mu' : (mu 𝕜 : IncidenceAlgebra 𝕜 α) = mu' 𝕜 := by
  classical
  exact left_inv_eq_right_inv (mu_mul_zeta _ _) (zeta_mul_mu' _ _)


lemma mu_eq_neg_sum_Ioc_of_ne (hab : a ≠ b) : mu 𝕜 a b = -∑ x ∈ Ioc a b, mu 𝕜 x b := by
  /-
    𝕜 : Type u_2
    α : Type u_5
    inst✝³ : Ring 𝕜
    inst✝² : PartialOrder α
    inst✝¹ : LocallyFiniteOrder α
    inst✝ : DecidableEq α
    a b : α
    hab : Ne a b
    ⊢ Eq ((IncidenceAlgebra.mu 𝕜) a b) (Neg.neg ((Finset.Ioc a b).sum fun x => (In …
  -/
  rw [mu_eq_mu', mu'_eq_sum_Ioc_of_ne hab]
  /-
    🎉 no goals
  -/


lemma zeta_mul_mu [DecidableRel (α := α) (· ≤ ·)] : (zeta 𝕜 * mu 𝕜 : IncidenceAlgebra 𝕜 α) = 1 := by
  /-
    𝕜 : Type u_2
    α : Type u_5
    inst✝⁴ : Ring 𝕜
    inst✝³ : PartialOrder α
    inst✝² : LocallyFiniteOrder α
    inst✝¹ : DecidableEq α
    inst✝ : DecidableRel fun x1 x2 => LE.le x1 x2
    ⊢ Eq (HMul.hMul (IncidenceAlgebra.zeta 𝕜) (IncidenceAlgebra.mu 𝕜)) 1
  -/
  rw [mu_eq_mu', zeta_mul_mu']
  /-
    🎉 no goals
  -/


lemma sum_Icc_mu_left (a b : α) : ∑ x ∈ Icc a b, mu 𝕜 x b = if a = b then 1 else 0 := by
  /-
    𝕜 : Type u_2
    α : Type u_5
    inst✝³ : Ring 𝕜
    inst✝² : PartialOrder α
    inst✝¹ : LocallyFiniteOrder α
    inst✝ : DecidableEq α
    a b : α
    ⊢ Eq ((Finset.Icc a b).sum fun x => (IncidenceAlgebra.mu 𝕜) x b) (ite (Eq a b) …
  -/
  rw [mu_eq_mu', sum_Icc_mu'_left]
  /-
    🎉 no goals
  -/


@[simp]
lemma mu_toDual (a b : α) : mu 𝕜 (toDual a) (toDual b) = mu 𝕜 b a := by
  /-
    𝕜 : Type u_2
    α : Type u_5
    inst✝³ : Ring 𝕜
    inst✝² : PartialOrder α
    inst✝¹ : LocallyFiniteOrder α
    inst✝ : DecidableEq α
    a b : α
    ⊢ Eq ((IncidenceAlgebra.mu 𝕜) (OrderDual.toDual a) (OrderDual.toDual b)) ((Inc …
  -/
  letI : DecidableRel (α := α) (· ≤ ·) := Classical.decRel _
  let mud : IncidenceAlgebra 𝕜 αᵒᵈ :=
    { toFun := fun a b ↦ mu 𝕜 (ofDual b) (ofDual a)
      eq_zero_of_not_le' := fun a b hab ↦ apply_eq_zero_of_not_le (by exact hab) _ }
  /-
    𝕜 : Type u_2
    α : Type u_5
    inst✝³ : Ring 𝕜
    inst✝² : PartialOrder α
    inst✝¹ : LocallyFiniteOrder α
    inst✝ : DecidableEq α
    a b : α
    this : DecidableRel fun x1 x2 => LE.le x1 x2 := Classical.decRel fun x1 x2 =>  …
    mud : IncidenceAlgebra 𝕜 (OrderDual α) := { toFun := fun a b => (IncidenceAlge …
    ⊢ Eq ((IncidenceAlgebra.mu 𝕜) (OrderDual.toDual a) (OrderDual.toDual b)) ((Inc …
  -/
  suffices mu 𝕜 = mud by rw [this]; rfl
  suffices mud * zeta 𝕜 = 1 by
    rw [← mu_mul_zeta] at this
    apply_fun (· * mu 𝕜) at this
    symm
    simpa [mul_assoc, zeta_mul_mu] using this
  /-
    𝕜 : Type u_2
    α : Type u_5
    inst✝³ : Ring 𝕜
    inst✝² : PartialOrder α
    inst✝¹ : LocallyFiniteOrder α
    inst✝ : DecidableEq α
    a b : α
    this : DecidableRel fun x1 x2 => LE.le x1 x2 := Classical.decRel fun x1 x2 =>  …
    mud : IncidenceAlgebra 𝕜 (OrderDual α) := { toFun := fun a b => (IncidenceAlge …
    ⊢ Eq (HMul.hMul mud (IncidenceAlgebra.zeta 𝕜)) 1
  -/
  clear a b
  /-
    𝕜 : Type u_2
    α : Type u_5
    inst✝³ : Ring 𝕜
    inst✝² : PartialOrder α
    inst✝¹ : LocallyFiniteOrder α
    inst✝ : DecidableEq α
    this : DecidableRel fun x1 x2 => LE.le x1 x2 := Classical.decRel fun x1 x2 =>  …
    mud : IncidenceAlgebra 𝕜 (OrderDual α) := { toFun := fun a b => (IncidenceAlge …
    ⊢ Eq (HMul.hMul mud (IncidenceAlgebra.zeta 𝕜)) 1
  -/
  ext a b
  /-
    case h
    𝕜 : Type u_2
    α : Type u_5
    inst✝³ : Ring 𝕜
    inst✝² : PartialOrder α
    inst✝¹ : LocallyFiniteOrder α
    inst✝ : DecidableEq α
    this : DecidableRel fun x1 x2 => LE.le x1 x2 := Classical.decRel fun x1 x2 =>  …
    mud : IncidenceAlgebra 𝕜 (OrderDual α) := { toFun := fun a b => (IncidenceAlge …
    a b : OrderDual α
    a✝ : LE.le a b
    ⊢ Eq ((HMul.hMul mud (IncidenceAlgebra.zeta 𝕜)) a b) (1 a b)
  -/
  simp only [mul_boole, one_apply, mul_apply, coe_mk, zeta_apply]
  calc
    ∑ x ∈ Icc a b, (if x ≤ b then mud a x else 0) = ∑ x ∈ Icc a b, mud a x := by
      congr! with x hx; exact if_pos (mem_Icc.1 hx).2
    _ = ∑ x ∈ Icc (ofDual b) (ofDual a), mu 𝕜 x (ofDual a) := by simp [Icc_orderDual_def, mud]
    _ = if ofDual b = ofDual a then 1 else 0 := sum_Icc_mu_left ..
    _ = if a = b then 1 else 0 := by simp [eq_comm]


@[simp] lemma mu_ofDual (a b : αᵒᵈ) : mu 𝕜 (ofDual a) (ofDual b) = mu 𝕜 b a := (mu_toDual ..).symm


/-- A general form of Möbius inversion. Based on lemma 2.1.2 of Incidence Algebras by Spiegel and
O'Donnell. -/
lemma moebius_inversion_top (f g : α → 𝕜) (h : ∀ x, g x = ∑ y ∈ Ici x, f y) (x : α) :
    f x = ∑ y ∈ Ici x, mu 𝕜 x y * g y := by
  /-
    𝕜 : Type u_2
    α : Type u_5
    inst✝⁴ : Ring 𝕜
    inst✝³ : PartialOrder α
    inst✝² : OrderTop α
    inst✝¹ : LocallyFiniteOrder α
    inst✝ : DecidableEq α
    f g : α → 𝕜
    h : ∀ (x : α), Eq (g x) ((Finset.Ici x).sum fun y => f y)
    x : α
    ⊢ Eq (f x) ((Finset.Ici x).sum fun y => HMul.hMul ((IncidenceAlgebra.mu 𝕜) x y …
  -/
  letI : DecidableRel (α := α) (· ≤ ·) := Classical.decRel _
  /-
    𝕜 : Type u_2
    α : Type u_5
    inst✝⁴ : Ring 𝕜
    inst✝³ : PartialOrder α
    inst✝² : OrderTop α
    inst✝¹ : LocallyFiniteOrder α
    inst✝ : DecidableEq α
    f g : α → 𝕜
    h : ∀ (x : α), Eq (g x) ((Finset.Ici x).sum fun y => f y)
    x : α
    this : DecidableRel fun x1 x2 => LE.le x1 x2 := Classical.decRel fun x1 x2 =>  …
    ⊢ Eq (f x) ((Finset.Ici x).sum fun y => HMul.hMul ((IncidenceAlgebra.mu 𝕜) x y …
  -/
  symm
  calc
    ∑ y ∈ Ici x, mu 𝕜 x y * g y = ∑ y ∈ Ici x, mu 𝕜 x y * ∑ z ∈ Ici y, f z := by simp_rw [h]
    _ = ∑ y ∈ Ici x, mu 𝕜 x y * ∑ z ∈ Ici y, zeta 𝕜 y z * f z := by
      congr with y
      rw [sum_congr rfl fun z hz ↦ ?_]
      rw [zeta_apply, if_pos (mem_Ici.mp ‹_›), one_mul]
    _ = ∑ y ∈ Ici x, ∑ z ∈ Ici y, mu 𝕜 x y * zeta 𝕜 y z * f z := by simp [mul_sum]
    _ = ∑ z ∈ Ici x, ∑ y ∈ Icc x z, mu 𝕜 x y * zeta 𝕜 y z * f z := by
      rw [sum_sigma' (Ici x) fun y ↦ Ici y]
      rw [sum_sigma' (Ici x) fun z ↦ Icc x z]
      simp only [mul_boole, MulZeroClass.zero_mul, ite_mul, zeta_apply]
      apply sum_nbij' (fun ⟨a, b⟩ ↦ ⟨b, a⟩) (fun ⟨a, b⟩ ↦ ⟨b, a⟩) <;>
        aesop (add simp mul_assoc) (add unsafe le_trans)
    _ = ∑ z ∈ Ici x, (mu 𝕜 * zeta 𝕜 : IncidenceAlgebra 𝕜 α) x z * f z := by
      simp_rw [mul_apply, sum_mul]
    _ = ∑ y ∈ Ici x, ∑ z ∈ Ici y, (1 : IncidenceAlgebra 𝕜 α) x z * f z := by
      simp [mu_mul_zeta 𝕜, ← add_sum_Ioi_eq_sum_Ici]
      exact sum_eq_zero fun y hy ↦ if_neg (mem_Ioi.mp hy).not_le
    _ = f x := by
      simp [one_apply, ← add_sum_Ioi_eq_sum_Ici]
      exact sum_eq_zero fun y hy ↦ if_neg (mem_Ioi.mp hy).not_le


/-- A general form of Möbius inversion. Based on lemma 2.1.3 of Incidence Algebras by Spiegel and
O'Donnell. -/
lemma moebius_inversion_bot (f g : α → 𝕜) (h : ∀ x, g x = ∑ y ∈ Iic x, f y) (x : α) :
    f x = ∑ y ∈ Iic x, mu 𝕜 y x * g y := by
  /-
    𝕜 : Type u_2
    α : Type u_5
    inst✝⁴ : Ring 𝕜
    inst✝³ : PartialOrder α
    inst✝² : OrderBot α
    inst✝¹ : LocallyFiniteOrder α
    inst✝ : DecidableEq α
    f g : α → 𝕜
    h : ∀ (x : α), Eq (g x) ((Finset.Iic x).sum fun y => f y)
    x : α
    ⊢ Eq (f x) ((Finset.Iic x).sum fun y => HMul.hMul ((IncidenceAlgebra.mu 𝕜) y x …
  -/
  convert moebius_inversion_top (α := αᵒᵈ) f g h x using 3; erw [mu_toDual]
                                                            /-
                                                              🎉 no goals
                                                            -/


lemma zeta_prod_apply (a b : α × β) : zeta 𝕜 a b = zeta 𝕜 a.1 b.1 * zeta 𝕜 a.2 b.2 := by
  /-
    𝕜 : Type u_2
    α : Type u_5
    β : Type u_6
    inst✝⁴ : Ring 𝕜
    inst✝³ : Preorder α
    inst✝² : Preorder β
    inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
    inst✝ : DecidableRel fun x1 x2 => LE.le x1 x2
    a b : Prod α β
    ⊢ Eq ((IncidenceAlgebra.zeta 𝕜) a b) (HMul.hMul ((IncidenceAlgebra.zeta 𝕜) a.1 …
  -/
  simp [← ite_and, Prod.le_def, and_comm]
  /-
    🎉 no goals
  -/


lemma zeta_prod_mk (a₁ a₂ : α) (b₁ b₂ : β) :
    zeta 𝕜 (a₁, b₁) (a₂, b₂) = zeta 𝕜 a₁ a₂ * zeta 𝕜 b₁ b₂ := zeta_prod_apply _ _ _


/-- The cartesian product of two incidence algebras. -/
protected def prod : IncidenceAlgebra 𝕜 (α × β) where
  toFun x y := f x.1 y.1 * g x.2 y.2
  eq_zero_of_not_le' x y hxy := by
    /-
      F : Type u_1
      𝕜 : Type u_2
      𝕝 : Type u_3
      𝕞 : Type u_4
      α : Type u_5
      β : Type u_6
      inst✝² : Ring 𝕜
      inst✝¹ : Preorder α
      inst✝ : Preorder β
      f f₁ f₂ : IncidenceAlgebra 𝕜 α
      g g₁ g₂ : IncidenceAlgebra 𝕜 β
      x y : Prod α β
      hxy : Not (LE.le x y)
      ⊢ Eq ((fun x y => HMul.hMul (f x.1 y.1) (g x.2 y.2)) x y) 0
    -/
    rw [Prod.le_def, not_and_or] at hxy
    /-
      F : Type u_1
      𝕜 : Type u_2
      𝕝 : Type u_3
      𝕞 : Type u_4
      α : Type u_5
      β : Type u_6
      inst✝² : Ring 𝕜
      inst✝¹ : Preorder α
      inst✝ : Preorder β
      f f₁ f₂ : IncidenceAlgebra 𝕜 α
      g g₁ g₂ : IncidenceAlgebra 𝕜 β
      x y : Prod α β
      hxy : Or (Not (LE.le x.1 y.1)) (Not (LE.le x.2 y.2))
      ⊢ Eq ((fun x y => HMul.hMul (f x.1 y.1) (g x.2 y.2)) x y) 0
    -/
                                /-
                                  🎉 no goals
                                -/
    obtain hxy | hxy := hxy <;> simp [apply_eq_zero_of_not_le hxy]
                                /-
                                  🎉 no goals
                                -/


lemma prod_mk (a₁ a₂ : α) (b₁ b₂ : β) : f.prod g (a₁, b₁) (a₂, b₂) = f a₁ a₂ * g b₁ b₂ := rfl

@[simp] lemma prod_apply (x y : α × β) : f.prod g x y = f x.1 y.1 * g x.2 y.2 := rfl


/-- This is a version of `IncidenceAlgebra.prod_mul_prod` that works over non-commutative rings. -/
lemma prod_mul_prod' [LocallyFiniteOrder α] [LocallyFiniteOrder β]
    [DecidableRel (α := α × β) (· ≤ ·)]
    (h : ∀ a₁ a₂ a₃ b₁ b₂ b₃,
        f₁ a₁ a₂ * g₁ b₁ b₂ * (f₂ a₂ a₃ * g₂ b₂ b₃) = f₁ a₁ a₂ * f₂ a₂ a₃ * (g₁ b₁ b₂ * g₂ b₂ b₃)) :
    f₁.prod g₁ * f₂.prod g₂ = (f₁ * f₂).prod (g₁ * g₂) := by
  /-
    𝕜 : Type u_2
    α : Type u_5
    β : Type u_6
    inst✝⁵ : Ring 𝕜
    inst✝⁴ : Preorder α
    inst✝³ : Preorder β
    f₁ f₂ : IncidenceAlgebra 𝕜 α
    g₁ g₂ : IncidenceAlgebra 𝕜 β
    inst✝² : LocallyFiniteOrder α
    inst✝¹ : LocallyFiniteOrder β
    inst✝ : DecidableRel fun x1 x2 => LE.le x1 x2
    h : ∀ (a₁ a₂ a₃ : α) (b₁ b₂ b₃ : β), Eq (HMul.hMul (HMul.hMul (f₁ a₁ a₂) (g₁ b …
    ⊢ Eq (HMul.hMul (f₁.prod g₁) (f₂.prod g₂)) ((HMul.hMul f₁ f₂).prod (HMul.hMul  …
  -/
  ext x y; simp [Icc_prod_def, sum_mul_sum, h, sum_product]
           /-
             🎉 no goals
           -/


@[simp]
lemma one_prod_one [DecidableEq α] [DecidableEq β] :
    (.prod 1 1 : IncidenceAlgebra 𝕜 (α × β)) = 1 := by
  /-
    𝕜 : Type u_2
    α : Type u_5
    β : Type u_6
    inst✝⁴ : Ring 𝕜
    inst✝³ : Preorder α
    inst✝² : Preorder β
    inst✝¹ : DecidableEq α
    inst✝ : DecidableEq β
    ⊢ Eq (IncidenceAlgebra.prod 1 1) 1
  -/
  ext x y; simp [Prod.ext_iff, ← ite_and, and_comm]
           /-
             🎉 no goals
           -/


@[simp]
lemma zeta_prod_zeta [DecidableRel (α := α) (· ≤ ·)] [DecidableRel (α := β) (· ≤ ·)] :
    (zeta 𝕜).prod (zeta 𝕜) = (zeta 𝕜 : IncidenceAlgebra 𝕜 (α × β)) := by
  /-
    𝕜 : Type u_2
    α : Type u_5
    β : Type u_6
    inst✝⁴ : Ring 𝕜
    inst✝³ : Preorder α
    inst✝² : Preorder β
    inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
    inst✝ : DecidableRel fun x1 x2 => LE.le x1 x2
    ⊢ Eq ((IncidenceAlgebra.zeta 𝕜).prod (IncidenceAlgebra.zeta 𝕜)) (IncidenceAlge …
  -/
  ext x y hxy; simp [hxy, hxy.1, hxy.2]
               /-
                 🎉 no goals
               -/


@[simp]
lemma prod_mul_prod : f₁.prod g₁ * f₂.prod g₂ = (f₁ * f₂).prod (g₁ * g₂) :=
  prod_mul_prod' _ _ _ _ fun _ _ _ _ _ _ ↦ mul_mul_mul_comm ..


/-- The Möbius function on a product order. Based on lemma 2.1.13 of Incidence Algebras by Spiegel
and O'Donnell. -/
@[simp]
lemma mu_prod_mu : (mu 𝕜).prod (mu 𝕜) = (mu 𝕜 : IncidenceAlgebra 𝕜 (α × β)) := by
  /-
    𝕜 : Type u_2
    α : Type u_5
    β : Type u_6
    inst✝⁸ : Ring 𝕜
    inst✝⁷ : PartialOrder α
    inst✝⁶ : PartialOrder β
    inst✝⁵ : LocallyFiniteOrder α
    inst✝⁴ : LocallyFiniteOrder β
    inst✝³ : DecidableEq α
    inst✝² : DecidableEq β
    inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
    inst✝ : DecidableRel fun x1 x2 => LE.le x1 x2
    ⊢ Eq ((IncidenceAlgebra.mu 𝕜).prod (IncidenceAlgebra.mu 𝕜)) (IncidenceAlgebra. …
  -/
  refine left_inv_eq_right_inv ?_ zeta_mul_mu
  /-
    𝕜 : Type u_2
    α : Type u_5
    β : Type u_6
    inst✝⁸ : Ring 𝕜
    inst✝⁷ : PartialOrder α
    inst✝⁶ : PartialOrder β
    inst✝⁵ : LocallyFiniteOrder α
    inst✝⁴ : LocallyFiniteOrder β
    inst✝³ : DecidableEq α
    inst✝² : DecidableEq β
    inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
    inst✝ : DecidableRel fun x1 x2 => LE.le x1 x2
    ⊢ Eq (HMul.hMul ((IncidenceAlgebra.mu 𝕜).prod (IncidenceAlgebra.mu 𝕜)) (Incide …
  -/
  rw [← zeta_prod_zeta, prod_mul_prod', mu_mul_zeta, mu_mul_zeta, one_prod_one]
  /-
    case h
    𝕜 : Type u_2
    α : Type u_5
    β : Type u_6
    inst✝⁸ : Ring 𝕜
    inst✝⁷ : PartialOrder α
    inst✝⁶ : PartialOrder β
    inst✝⁵ : LocallyFiniteOrder α
    inst✝⁴ : LocallyFiniteOrder β
    inst✝³ : DecidableEq α
    inst✝² : DecidableEq β
    inst✝¹ : DecidableRel fun x1 x2 => LE.le x1 x2
    inst✝ : DecidableRel fun x1 x2 => LE.le x1 x2
    ⊢ ∀ (a₁ a₂ a₃ : α) (b₁ b₂ b₃ : β), Eq (HMul.hMul (HMul.hMul ((IncidenceAlgebra …
  -/
  exact fun _ _ _ _ _ _ ↦ Commute.mul_mul_mul_comm (by simp : _ = _) _ _
  /-
    🎉 no goals
  -/


