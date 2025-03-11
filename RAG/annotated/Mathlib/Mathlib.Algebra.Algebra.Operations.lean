theorem algebraMap_mem (r : R) : algebraMap R A r ∈ (1 : SubMulAction R A) :=
  ⟨r, (algebraMap_eq_smul_one r).symm⟩


theorem mem_one' {x : A} : x ∈ (1 : SubMulAction R A) ↔ ∃ y, algebraMap R A y = x :=
                           /-
                             R : Type u
                             A : Type v
                             inst✝² : CommSemiring R
                             inst✝¹ : Semiring A
                             inst✝ : Algebra R A
                             x : A
                             r : R
                             ⊢ Iff (Eq ((fun r => HSMul.hSMul r 1) r) x) (Eq ((algebraMap R A) r) x)
                           -/
  exists_congr fun r => by rw [algebraMap_eq_smul_one]
                           /-
                             🎉 no goals
                           -/


/-- `1 : Submodule R A` is the submodule `R ∙ 1` of A.
TODO: potentially change this back to `LinearMap.range (Algebra.linearMap R A)`
once a version of `Algebra` without the `commutes'` field is introduced.
See issue https://github.com/leanprover-community/mathlib4/issues/18110.
-/
instance one : One (Submodule R A) :=
  ⟨LinearMap.range (LinearMap.toSpanSingleton R A 1)⟩


theorem one_eq_span : (1 : Submodule R A) = R ∙ 1 :=
  (LinearMap.span_singleton_eq_range _ _ _).symm


theorem le_one_toAddSubmonoid : 1 ≤ (1 : Submodule R A).toAddSubmonoid := by
  /-
    R : Type u
    inst✝² : Semiring R
    A : Type v
    inst✝¹ : Semiring A
    inst✝ : Module R A
    ⊢ LE.le 1 (Submodule.toAddSubmonoid 1)
  -/
  rintro x ⟨n, rfl⟩
  /-
    case intro
    R : Type u
    inst✝² : Semiring R
    A : Type v
    inst✝¹ : Semiring A
    inst✝ : Module R A
    n : Nat
    ⊢ Membership.mem (Submodule.toAddSubmonoid 1) ((Nat.castAddMonoidHom A) n)
  -/
  exact ⟨n, show (n : R) • (1 : A) = n by rw [Nat.cast_smul_eq_nsmul, nsmul_one]⟩
  /-
    🎉 no goals
  -/


@[simp]
theorem toSubMulAction_one : (1 : Submodule R A).toSubMulAction = 1 :=
                         /-
                           R : Type u
                           inst✝² : Semiring R
                           A : Type v
                           inst✝¹ : Semiring A
                           inst✝ : Module R A
                           x✝ : A
                           ⊢ Iff (Membership.mem (Submodule.toSubMulAction 1) x✝) (Membership.mem 1 x✝)
                         -/
  SetLike.ext fun _ ↦ by rw [one_eq_span, SubMulAction.mem_one]; exact mem_span_singleton
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


theorem one_eq_span_one_set : (1 : Submodule R A) = span R 1 :=
  one_eq_span


@[simp]
theorem one_le {P : Submodule R A} : (1 : Submodule R A) ≤ P ↔ (1 : A) ∈ P := by
  -- Porting note: simpa no longer closes refl goals, so added `SetLike.mem_coe`
  /-
    R : Type u
    inst✝² : Semiring R
    A : Type v
    inst✝¹ : Semiring A
    inst✝ : Module R A
    P : Submodule R A
    ⊢ Iff (LE.le 1 P) (Membership.mem P 1)
  -/
  simp only [one_eq_span, span_le, Set.singleton_subset_iff, SetLike.mem_coe]
  /-
    🎉 no goals
  -/


instance : SMul (Submodule R A) (Submodule R M) where
  smul A' M' :=
  { __ := A'.toAddSubmonoid • M'.toAddSubmonoid
    smul_mem' := fun r m hm ↦ AddSubmonoid.smul_induction_on hm
                          /-
                            R : Type u
                            inst✝⁶ : Semiring R
                            A : Type v
                            inst✝⁵ : Semiring A
                            inst✝⁴ : Module R A
                            M : Type u_1
                            inst✝³ : AddCommMonoid M
                            inst✝² : Module R M
                            inst✝¹ : Module A M
                            inst✝ : IsScalarTower R A M
                            A' : Submodule R A
                            M' : Submodule R M
                            r : R
                            m✝ : M
                            hm✝ : Membership.mem __spread✝⁻⁰.carrier m✝
                            a : A
                            ha : Membership.mem A'.toAddSubmonoid a
                            m : M
                            hm : Membership.mem M'.toAddSubmonoid m
                            ⊢ Membership.mem __spread✝⁻⁰.carrier (HSMul.hSMul r (HSMul.hSMul a m))
                          -/
      (fun a ha m hm ↦ by rw [← smul_assoc]; exact AddSubmonoid.smul_mem_smul (A'.smul_mem r ha) hm)
                                             /-
                                               🎉 no goals
                                             -/
                           /-
                             R : Type u
                             inst✝⁶ : Semiring R
                             A : Type v
                             inst✝⁵ : Semiring A
                             inst✝⁴ : Module R A
                             M : Type u_1
                             inst✝³ : AddCommMonoid M
                             inst✝² : Module R M
                             inst✝¹ : Module A M
                             inst✝ : IsScalarTower R A M
                             A' : Submodule R A
                             M' : Submodule R M
                             r : R
                             m : M
                             hm : Membership.mem __spread✝⁻⁰.carrier m
                             m₁ m₂ : M
                             h₁ : Membership.mem __spread✝⁻⁰.carrier (HSMul.hSMul r m₁)
                             h₂ : Membership.mem __spread✝⁻⁰.carrier (HSMul.hSMul r m₂)
                             ⊢ Membership.mem __spread✝⁻⁰.carrier (HSMul.hSMul r (HAdd.hAdd m₁ m₂))
                           -/
      fun m₁ m₂ h₁ h₂ ↦ by rw [smul_add]; exact (A'.1 • M'.1).add_mem h₁ h₂ }
                                          /-
                                            🎉 no goals
                                          -/


theorem smul_toAddSubmonoid : (I • N).toAddSubmonoid = I.toAddSubmonoid • N.toAddSubmonoid := rfl


theorem smul_mem_smul {r} {n} (hr : r ∈ I) (hn : n ∈ N) : r • n ∈ I • N :=
  AddSubmonoid.smul_mem_smul hr hn


theorem smul_le : I • N ≤ P ↔ ∀ r ∈ I, ∀ n ∈ N, r • n ∈ P :=
  AddSubmonoid.smul_le


@[simp, norm_cast]
lemma coe_set_smul : (I : Set A) • N = I • N :=
  set_smul_eq_of_le _ _ _
    (fun _ _ hr hx ↦ smul_mem_smul hr hx)
    (smul_le.mpr fun _ hr _ hx ↦ mem_set_smul_of_mem_mem hr hx)


@[elab_as_elim]
theorem smul_induction_on {p : M → Prop} {x} (H : x ∈ I • N) (smul : ∀ r ∈ I, ∀ n ∈ N, p (r • n))
    (add : ∀ x y, p x → p y → p (x + y)) : p x :=
  AddSubmonoid.smul_induction_on H smul add


/-- Dependent version of `Submodule.smul_induction_on`. -/
@[elab_as_elim]
theorem smul_induction_on' {x : M} (hx : x ∈ I • N) {p : ∀ x, x ∈ I • N → Prop}
    (smul : ∀ (r : A) (hr : r ∈ I) (n : M) (hn : n ∈ N), p (r • n) (smul_mem_smul hr hn))
    (add : ∀ x hx y hy, p x hx → p y hy → p (x + y) (add_mem ‹_› ‹_›)) : p x hx := by
  /-
    R : Type u
    inst✝⁶ : Semiring R
    A : Type v
    inst✝⁵ : Semiring A
    inst✝⁴ : Module R A
    M : Type u_1
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    inst✝¹ : Module A M
    inst✝ : IsScalarTower R A M
    I : Submodule R A
    N : Submodule R M
    x : M
    hx : Membership.mem (HSMul.hSMul I N) x
    p : (x : M) → Membership.mem (HSMul.hSMul I N) x → Prop
    smul : ∀ (r : A) (hr : Membership.mem I r) (n : M) (hn : Membership.mem N n),  …
    add : ∀ (x : M) (hx : Membership.mem (HSMul.hSMul I N) x) (y : M) (hy : Member …
    ⊢ p x hx
  -/
  refine Exists.elim ?_ fun (h : x ∈ I • N) (H : p x h) ↦ H
  exact smul_induction_on hx (fun a ha x hx ↦ ⟨_, smul _ ha _ hx⟩)
    fun x y ⟨_, hx⟩ ⟨_, hy⟩ ↦ ⟨_, add _ _ _ _ hx hy⟩


theorem smul_mono (hij : I ≤ J) (hnp : N ≤ P) : I • N ≤ J • P :=
  AddSubmonoid.smul_le_smul hij hnp


theorem smul_mono_left (h : I ≤ J) : I • N ≤ J • N :=
  smul_mono h le_rfl


instance : CovariantClass (Submodule R A) (Submodule R M) HSMul.hSMul LE.le :=
  ⟨fun _ _ => smul_mono le_rfl⟩


@[deprecated smul_mono_right (since := "2024-03-31")]
protected theorem smul_mono_right (h : N ≤ P) : I • N ≤ I • P :=
  _root_.smul_mono_right I h


@[simp]
theorem smul_bot : I • (⊥ : Submodule R M) = ⊥ :=
  toAddSubmonoid_injective <| AddSubmonoid.addSubmonoid_smul_bot _


@[simp]
theorem bot_smul : (⊥ : Submodule R A) • N = ⊥ :=
                                     /-
                                       R : Type u
                                       inst✝⁶ : Semiring R
                                       A : Type v
                                       inst✝⁵ : Semiring A
                                       inst✝⁴ : Module R A
                                       M : Type u_1
                                       inst✝³ : AddCommMonoid M
                                       inst✝² : Module R M
                                       inst✝¹ : Module A M
                                       inst✝ : IsScalarTower R A M
                                       N : Submodule R M
                                       ⊢ ∀ (r : A), Membership.mem Bot.bot r → ∀ (n : M), Membership.mem N n → Member …
                                     -/
  le_bot_iff.mp <| smul_le.mpr <| by rintro _ rfl _ _; rw [zero_smul]; exact zero_mem _
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


theorem smul_sup : I • (N ⊔ P) = I • N ⊔ I • P :=
  toAddSubmonoid_injective <| by
    /-
      R : Type u
      inst✝⁶ : Semiring R
      A : Type v
      inst✝⁵ : Semiring A
      inst✝⁴ : Module R A
      M : Type u_1
      inst✝³ : AddCommMonoid M
      inst✝² : Module R M
      inst✝¹ : Module A M
      inst✝ : IsScalarTower R A M
      I : Submodule R A
      N P : Submodule R M
      ⊢ Eq (HSMul.hSMul I (Max.max N P)).toAddSubmonoid (Max.max (HSMul.hSMul I N) ( …
    -/
    simp only [smul_toAddSubmonoid, sup_toAddSubmonoid, AddSubmonoid.addSubmonoid_smul_sup]
    /-
      🎉 no goals
    -/


theorem sup_smul : (I ⊔ J) • N = I • N ⊔ J • N :=
  le_antisymm (smul_le.mpr fun mn hmn p hp ↦ by
    /-
      R : Type u
      inst✝⁶ : Semiring R
      A : Type v
      inst✝⁵ : Semiring A
      inst✝⁴ : Module R A
      M : Type u_1
      inst✝³ : AddCommMonoid M
      inst✝² : Module R M
      inst✝¹ : Module A M
      inst✝ : IsScalarTower R A M
      I J : Submodule R A
      N : Submodule R M
      mn : A
      hmn : Membership.mem (Max.max I J) mn
      p : M
      hp : Membership.mem N p
      ⊢ Membership.mem (Max.max (HSMul.hSMul I N) (HSMul.hSMul J N)) (HSMul.hSMul mn …
    -/
    obtain ⟨m, hm, n, hn, rfl⟩ := mem_sup.mp hmn
    /-
      case intro.intro.intro.intro
      R : Type u
      inst✝⁶ : Semiring R
      A : Type v
      inst✝⁵ : Semiring A
      inst✝⁴ : Module R A
      M : Type u_1
      inst✝³ : AddCommMonoid M
      inst✝² : Module R M
      inst✝¹ : Module A M
      inst✝ : IsScalarTower R A M
      I J : Submodule R A
      N : Submodule R M
      p : M
      hp : Membership.mem N p
      m : A
      hm : Membership.mem I m
      n : A
      hn : Membership.mem J n
      hmn : Membership.mem (Max.max I J) (HAdd.hAdd m n)
      ⊢ Membership.mem (Max.max (HSMul.hSMul I N) (HSMul.hSMul J N)) (HSMul.hSMul (H …
    -/
    rw [add_smul]; exact add_mem_sup (smul_mem_smul hm hp) <| smul_mem_smul hn hp)
                   /-
                     🎉 no goals
                   -/
    (sup_le (smul_mono_left le_sup_left) <| smul_mono_left le_sup_right)


protected theorem smul_assoc {B} [Semiring B] [Module R B] [Module A B] [Module B M]
    [IsScalarTower R A B] [IsScalarTower R B M] [IsScalarTower A B M]
    (I : Submodule R A) (J : Submodule R B) (N : Submodule R M) :
    (I • J) • N = I • J • N :=
  le_antisymm
    (smul_le.2 fun _ hrsij t htn ↦ smul_induction_on hrsij
      (fun r hr s hs ↦ smul_assoc r s t ▸ smul_mem_smul hr (smul_mem_smul hs htn))
      fun x y ↦ (add_smul x y t).symm ▸ add_mem)
    (smul_le.2 fun r hr _ hsn ↦ smul_induction_on hsn
      (fun j hj n hn ↦ (smul_assoc r j n).symm ▸ smul_mem_smul (smul_mem_smul hr hj) hn)
      fun m₁ m₂ ↦ (smul_add r m₁ m₂) ▸ add_mem)


@[deprecated smul_inf_le (since := "2024-03-31")]
protected theorem smul_inf_le (M₁ M₂ : Submodule R M) :
    I • (M₁ ⊓ M₂) ≤ I • M₁ ⊓ I • M₂ := smul_inf_le _ _ _


theorem smul_iSup {ι : Sort*} {I : Submodule R A} {t : ι → Submodule R M} :
    I • (⨆ i, t i)= ⨆ i, I • t i :=
  toAddSubmonoid_injective <| by
    /-
      R : Type u
      inst✝⁶ : Semiring R
      A : Type v
      inst✝⁵ : Semiring A
      inst✝⁴ : Module R A
      M : Type u_1
      inst✝³ : AddCommMonoid M
      inst✝² : Module R M
      inst✝¹ : Module A M
      inst✝ : IsScalarTower R A M
      ι : Sort u_2
      I : Submodule R A
      t : ι → Submodule R M
      ⊢ Eq (HSMul.hSMul I (iSup fun i => t i)).toAddSubmonoid (iSup fun i => HSMul.h …
    -/
    simp only [smul_toAddSubmonoid, iSup_toAddSubmonoid, AddSubmonoid.smul_iSup]
    /-
      🎉 no goals
    -/


theorem iSup_smul {ι : Sort*} {t : ι → Submodule R A} {N : Submodule R M} :
    (⨆ i, t i) • N = ⨆ i, t i • N :=
  le_antisymm (smul_le.mpr fun t ht s hs ↦ iSup_induction _ (C := (· • s ∈ _)) ht
    (fun i t ht ↦ mem_iSup_of_mem i <| smul_mem_smul ht hs)
        /-
          R : Type u
          inst✝⁶ : Semiring R
          A : Type v
          inst✝⁵ : Semiring A
          inst✝⁴ : Module R A
          M : Type u_1
          inst✝³ : AddCommMonoid M
          inst✝² : Module R M
          inst✝¹ : Module A M
          inst✝ : IsScalarTower R A M
          ι : Sort u_2
          t✝ : ι → Submodule R A
          N : Submodule R M
          t : A
          ht : Membership.mem (iSup fun i => t✝ i) t
          s : M
          hs : Membership.mem N s
          ⊢ (fun x => Membership.mem (iSup fun i => HSMul.hSMul (t✝ i) N) (HSMul.hSMul x …
        -/
                             /-
                               🎉 no goals
                             -/
    (by simp_rw [zero_smul]; apply zero_mem) fun x y ↦ by simp_rw [add_smul]; apply add_mem)
                                                                              /-
                                                                                🎉 no goals
                                                                              -/
    (iSup_le fun i ↦ Submodule.smul_mono_left <| le_iSup _ i)


@[deprecated smul_iInf_le (since := "2024-03-31")]
protected theorem smul_iInf_le {ι : Sort*} {I : Submodule R A} {t : ι → Submodule R M} :
    I • iInf t ≤ ⨅ i, I • t i :=
  smul_iInf_le


protected theorem one_smul : (1 : Submodule R A) • N = N := by
  /-
    R : Type u
    inst✝⁶ : Semiring R
    A : Type v
    inst✝⁵ : Semiring A
    inst✝⁴ : Module R A
    M : Type u_1
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    inst✝¹ : Module A M
    inst✝ : IsScalarTower R A M
    N : Submodule R M
    ⊢ Eq (HSMul.hSMul 1 N) N
  -/
  refine le_antisymm (smul_le.mpr fun r hr m hm ↦ ?_) fun m hm ↦ ?_
    /-
      case refine_1
      R : Type u
      inst✝⁶ : Semiring R
      A : Type v
      inst✝⁵ : Semiring A
      inst✝⁴ : Module R A
      M : Type u_1
      inst✝³ : AddCommMonoid M
      inst✝² : Module R M
      inst✝¹ : Module A M
      inst✝ : IsScalarTower R A M
      N : Submodule R M
      r : A
      hr : Membership.mem 1 r
      m : M
      hm : Membership.mem N m
      ⊢ Membership.mem N (HSMul.hSMul r m)
    -/
  · obtain ⟨r, rfl⟩ := hr
    /-
      case refine_1.intro
      R : Type u
      inst✝⁶ : Semiring R
      A : Type v
      inst✝⁵ : Semiring A
      inst✝⁴ : Module R A
      M : Type u_1
      inst✝³ : AddCommMonoid M
      inst✝² : Module R M
      inst✝¹ : Module A M
      inst✝ : IsScalarTower R A M
      N : Submodule R M
      m : M
      hm : Membership.mem N m
      r : R
      ⊢ Membership.mem N (HSMul.hSMul ((LinearMap.toSpanSingleton R A 1) r) m)
    -/
    rw [LinearMap.toSpanSingleton_apply, smul_one_smul]; exact N.smul_mem r hm
                                                         /-
                                                           🎉 no goals
                                                         -/
    /-
      case refine_2
      R : Type u
      inst✝⁶ : Semiring R
      A : Type v
      inst✝⁵ : Semiring A
      inst✝⁴ : Module R A
      M : Type u_1
      inst✝³ : AddCommMonoid M
      inst✝² : Module R M
      inst✝¹ : Module A M
      inst✝ : IsScalarTower R A M
      N : Submodule R M
      m : M
      hm : Membership.mem N m
      ⊢ Membership.mem (HSMul.hSMul 1 N) m
    -/
  · rw [← one_smul A m]; exact smul_mem_smul (one_le.mp le_rfl) hm
                         /-
                           🎉 no goals
                         -/


theorem smul_subset_smul : (↑I : Set A) • (↑N : Set M) ⊆ (↑(I • N) : Set M) :=
  AddSubmonoid.smul_subset_smul


/-- Multiplication of sub-R-modules of an R-module A that is also a semiring. The submodule `M * N`
consists of finite sums of elements `m * n` for `m ∈ M` and `n ∈ N`. -/
instance mul : Mul (Submodule R A) where
  mul := (· • ·)


theorem mul_mem_mul (hm : m ∈ M) (hn : n ∈ N) : m * n ∈ M * N :=
  smul_mem_smul hm hn


theorem mul_le : M * N ≤ P ↔ ∀ m ∈ M, ∀ n ∈ N, m * n ∈ P :=
  smul_le


theorem mul_toAddSubmonoid (M N : Submodule R A) :
    (M * N).toAddSubmonoid = M.toAddSubmonoid * N.toAddSubmonoid := rfl


@[elab_as_elim]
protected theorem mul_induction_on {C : A → Prop} {r : A} (hr : r ∈ M * N)
    (hm : ∀ m ∈ M, ∀ n ∈ N, C (m * n)) (ha : ∀ x y, C x → C y → C (x + y)) : C r :=
  smul_induction_on hr hm ha


/-- A dependent version of `mul_induction_on`. -/
@[elab_as_elim]
protected theorem mul_induction_on' {C : ∀ r, r ∈ M * N → Prop}
    (mem_mul_mem : ∀ m (hm : m ∈ M) n (hn : n ∈ N), C (m * n) (mul_mem_mul hm hn))
    (add : ∀ x hx y hy, C x hx → C y hy → C (x + y) (add_mem hx hy)) {r : A} (hr : r ∈ M * N) :
    C r hr :=
  smul_induction_on' hr mem_mul_mem add


@[simp]
theorem mul_bot : M * ⊥ = ⊥ :=
  smul_bot _


@[simp]
theorem bot_mul : ⊥ * M = ⊥ :=
  bot_smul _


protected theorem one_mul : (1 : Submodule R A) * M = M :=
  Submodule.one_smul _


@[mono]
theorem mul_le_mul (hmp : M ≤ P) (hnq : N ≤ Q) : M * N ≤ P * Q :=
  smul_mono hmp hnq


theorem mul_le_mul_left (h : M ≤ N) : M * P ≤ N * P :=
  smul_mono_left h


theorem mul_le_mul_right (h : N ≤ P) : M * N ≤ M * P :=
  smul_mono_right _ h


theorem mul_comm_of_commute (h : ∀ m ∈ M, ∀ n ∈ N, Commute m n) : M * N = N * M :=
  toAddSubmonoid_injective <| AddSubmonoid.mul_comm_of_commute h


theorem mul_sup : M * (N ⊔ P) = M * N ⊔ M * P :=
  smul_sup _ _ _


theorem sup_mul : (M ⊔ N) * P = M * P ⊔ N * P :=
  sup_smul _ _ _


theorem mul_subset_mul : (↑M : Set A) * (↑N : Set A) ⊆ (↑(M * N) : Set A) :=
  smul_subset_smul _ _


lemma restrictScalars_mul {A B C} [Semiring A] [Semiring B] [Semiring C]
    [SMul A B] [Module A C] [Module B C] [IsScalarTower A C C] [IsScalarTower B C C]
    [IsScalarTower A B C] {I J : Submodule B C} :
    (I * J).restrictScalars A = I.restrictScalars A * J.restrictScalars A :=
  rfl


theorem iSup_mul (s : ι → Submodule R A) (t : Submodule R A) : (⨆ i, s i) * t = ⨆ i, s i * t :=
  iSup_smul


theorem mul_iSup (t : Submodule R A) (s : ι → Submodule R A) : (t * ⨆ i, s i) = ⨆ i, t * s i :=
  smul_iSup


/-- Sub-`R`-modules of an `R`-module form an idempotent semiring. -/
instance : NonUnitalSemiring (Submodule R A) where
  __ := toAddSubmonoid_injective.semigroup _ mul_toAddSubmonoid
  zero_mul := bot_mul
  mul_zero := mul_bot
  left_distrib := mul_sup
  right_distrib := sup_mul


instance : Pow (Submodule R A) ℕ where
  pow s n := npowRec n s


theorem pow_eq_npowRec {n : ℕ} : M ^ n = npowRec n M := rfl


protected theorem pow_zero : M ^ 0 = 1 := rfl


protected theorem pow_succ {n : ℕ} : M ^ (n + 1) = M ^ n * M := rfl


protected theorem pow_one : M ^ 1 = M := by
  /-
    R : Type u
    inst✝³ : Semiring R
    A : Type v
    inst✝² : Semiring A
    inst✝¹ : Module R A
    inst✝ : IsScalarTower R A A
    M : Submodule R A
    ⊢ Eq (HPow.hPow M 1) M
  -/
  rw [Submodule.pow_succ, Submodule.pow_zero, Submodule.one_mul]
  /-
    🎉 no goals
  -/


theorem pow_toAddSubmonoid {n : ℕ} (h : n ≠ 0) : (M ^ n).toAddSubmonoid = M.toAddSubmonoid ^ n := by
  induction n with
  | zero => exact (h rfl).elim
  | succ n ih =>
    rw [Submodule.pow_succ, pow_succ, mul_toAddSubmonoid]
    cases n with
    | zero => rw [Submodule.pow_zero, pow_zero, one_mul, ← mul_toAddSubmonoid, Submodule.one_mul]
    | succ n => rw [ih n.succ_ne_zero]


theorem le_pow_toAddSubmonoid {n : ℕ} : M.toAddSubmonoid ^ n ≤ (M ^ n).toAddSubmonoid := by
  /-
    R : Type u
    inst✝³ : Semiring R
    A : Type v
    inst✝² : Semiring A
    inst✝¹ : Module R A
    inst✝ : IsScalarTower R A A
    M : Submodule R A
    n : Nat
    ⊢ LE.le (HPow.hPow M.toAddSubmonoid n) (HPow.hPow M n).toAddSubmonoid
  -/
  obtain rfl | hn := Decidable.eq_or_ne n 0
    /-
      case inl
      R : Type u
      inst✝³ : Semiring R
      A : Type v
      inst✝² : Semiring A
      inst✝¹ : Module R A
      inst✝ : IsScalarTower R A A
      M : Submodule R A
      ⊢ LE.le (HPow.hPow M.toAddSubmonoid 0) (HPow.hPow M 0).toAddSubmonoid
    -/
  · rw [Submodule.pow_zero, pow_zero]
    /-
      case inl
      R : Type u
      inst✝³ : Semiring R
      A : Type v
      inst✝² : Semiring A
      inst✝¹ : Module R A
      inst✝ : IsScalarTower R A A
      M : Submodule R A
      ⊢ LE.le 1 (Submodule.toAddSubmonoid 1)
    -/
    exact le_one_toAddSubmonoid
    /-
      🎉 no goals
    -/
    /-
      case inr
      R : Type u
      inst✝³ : Semiring R
      A : Type v
      inst✝² : Semiring A
      inst✝¹ : Module R A
      inst✝ : IsScalarTower R A A
      M : Submodule R A
      n : Nat
      hn : Ne n 0
      ⊢ LE.le (HPow.hPow M.toAddSubmonoid n) (HPow.hPow M n).toAddSubmonoid
    -/
  · exact (pow_toAddSubmonoid M hn).ge
    /-
      🎉 no goals
    -/


theorem pow_subset_pow {n : ℕ} : (↑M : Set A) ^ n ⊆ ↑(M ^ n : Submodule R A) :=
  trans AddSubmonoid.pow_subset_pow (le_pow_toAddSubmonoid M)


theorem pow_mem_pow {x : A} (hx : x ∈ M) (n : ℕ) : x ^ n ∈ M ^ n :=
  pow_subset_pow _ <| Set.pow_mem_pow hx


theorem one_eq_range : (1 : Submodule R A) = LinearMap.range (Algebra.linearMap R A) := by
  rw [one_eq_span, LinearMap.span_singleton_eq_range,
    LinearMap.toSpanSingleton_eq_algebra_linearMap]


theorem algebraMap_mem (r : R) : algebraMap R A r ∈ (1 : Submodule R A) := by
  /-
    R : Type u
    inst✝² : CommSemiring R
    A : Type v
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    r : R
    ⊢ Membership.mem 1 ((algebraMap R A) r)
  -/
  rw [one_eq_range]; exact LinearMap.mem_range_self _ _
                     /-
                       🎉 no goals
                     -/


@[simp]
theorem mem_one {x : A} : x ∈ (1 : Submodule R A) ↔ ∃ y, algebraMap R A y = x := by
  /-
    R : Type u
    inst✝² : CommSemiring R
    A : Type v
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    x : A
    ⊢ Iff (Membership.mem 1 x) (Exists fun y => Eq ((algebraMap R A) y) x)
  -/
  rw [one_eq_range]; rfl
                     /-
                       🎉 no goals
                     -/


protected theorem map_one {A'} [Semiring A'] [Algebra R A'] (f : A →ₐ[R] A') :
    map f.toLinearMap (1 : Submodule R A) = 1 := by
  /-
    R : Type u
    inst✝⁴ : CommSemiring R
    A : Type v
    inst✝³ : Semiring A
    inst✝² : Algebra R A
    A' : Type u_1
    inst✝¹ : Semiring A'
    inst✝ : Algebra R A'
    f : AlgHom R A A'
    ⊢ Eq (Submodule.map f.toLinearMap 1) 1
  -/
  ext
  /-
    case h
    R : Type u
    inst✝⁴ : CommSemiring R
    A : Type v
    inst✝³ : Semiring A
    inst✝² : Algebra R A
    A' : Type u_1
    inst✝¹ : Semiring A'
    inst✝ : Algebra R A'
    f : AlgHom R A A'
    x✝ : A'
    ⊢ Iff (Membership.mem (Submodule.map f.toLinearMap 1) x✝) (Membership.mem 1 x✝)
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem map_op_one :
    map (↑(opLinearEquiv R : A ≃ₗ[R] Aᵐᵒᵖ) : A →ₗ[R] Aᵐᵒᵖ) (1 : Submodule R A) = 1 := by
  /-
    R : Type u
    inst✝² : CommSemiring R
    A : Type v
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    ⊢ Eq (Submodule.map (↑(MulOpposite.opLinearEquiv R)) 1) 1
  -/
  ext x
  /-
    case h
    R : Type u
    inst✝² : CommSemiring R
    A : Type v
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    x : MulOpposite A
    ⊢ Iff (Membership.mem (Submodule.map (↑(MulOpposite.opLinearEquiv R)) 1) x) (M …
  -/
  induction x
  /-
    case h.h
    R : Type u
    inst✝² : CommSemiring R
    A : Type v
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    X✝ : A
    ⊢ Iff (Membership.mem (Submodule.map (↑(MulOpposite.opLinearEquiv R)) 1) (MulO …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem comap_op_one :
    comap (↑(opLinearEquiv R : A ≃ₗ[R] Aᵐᵒᵖ) : A →ₗ[R] Aᵐᵒᵖ) (1 : Submodule R Aᵐᵒᵖ) = 1 := by
  /-
    R : Type u
    inst✝² : CommSemiring R
    A : Type v
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    ⊢ Eq (Submodule.comap (↑(MulOpposite.opLinearEquiv R)) 1) 1
  -/
  ext
  /-
    case h
    R : Type u
    inst✝² : CommSemiring R
    A : Type v
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    x✝ : A
    ⊢ Iff (Membership.mem (Submodule.comap (↑(MulOpposite.opLinearEquiv R)) 1) x✝) …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem map_unop_one :
    map (↑(opLinearEquiv R : A ≃ₗ[R] Aᵐᵒᵖ).symm : Aᵐᵒᵖ →ₗ[R] A) (1 : Submodule R Aᵐᵒᵖ) = 1 := by
  /-
    R : Type u
    inst✝² : CommSemiring R
    A : Type v
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    ⊢ Eq (Submodule.map (↑(MulOpposite.opLinearEquiv R).symm) 1) 1
  -/
  rw [← comap_equiv_eq_map_symm, comap_op_one]
  /-
    🎉 no goals
  -/


@[simp]
theorem comap_unop_one :
    comap (↑(opLinearEquiv R : A ≃ₗ[R] Aᵐᵒᵖ).symm : Aᵐᵒᵖ →ₗ[R] A) (1 : Submodule R A) = 1 := by
  /-
    R : Type u
    inst✝² : CommSemiring R
    A : Type v
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    ⊢ Eq (Submodule.comap (↑(MulOpposite.opLinearEquiv R).symm) 1) 1
  -/
  rw [← map_equiv_eq_comap_symm, map_op_one]
  /-
    🎉 no goals
  -/


theorem mul_eq_map₂ : M * N = map₂ (LinearMap.mul R A) M N :=
  le_antisymm (mul_le.mpr fun _m hm _n ↦ apply_mem_map₂ _ hm)
    (map₂_le.mpr fun _m hm _n ↦ mul_mem_mul hm)


theorem span_mul_span : span R S * span R T = span R (S * T) := by
  /-
    R : Type u
    inst✝² : CommSemiring R
    A : Type v
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    S T : Set A
    ⊢ Eq (HMul.hMul (Submodule.span R S) (Submodule.span R T)) (Submodule.span R ( …
  -/
  rw [mul_eq_map₂]; apply map₂_span_span
                    /-
                      🎉 no goals
                    -/


                                                     /-
                                                       R : Type u
                                                       inst✝² : CommSemiring R
                                                       A : Type v
                                                       inst✝¹ : Semiring A
                                                       inst✝ : Algebra R A
                                                       M N : Submodule R A
                                                       ⊢ Eq (HMul.hMul M N) (Submodule.span R (HMul.hMul ↑M ↑N))
                                                     -/
lemma mul_def : M * N = span R (M * N : Set A) := by simp [← span_mul_span]
                                                     /-
                                                       🎉 no goals
                                                     -/


protected theorem mul_one : M * 1 = M := by
  /-
    R : Type u
    inst✝² : CommSemiring R
    A : Type v
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    M : Submodule R A
    ⊢ Eq (HMul.hMul M 1) M
  -/
  conv_lhs => rw [one_eq_span, ← span_eq M]
  /-
    R : Type u
    inst✝² : CommSemiring R
    A : Type v
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    M : Submodule R A
    ⊢ Eq (HMul.hMul (Submodule.span R ↑M) (Submodule.span R (Singleton.singleton 1 …
  -/
  erw [span_mul_span, mul_one, span_eq]
  /-
    🎉 no goals
  -/


protected theorem map_mul {A'} [Semiring A'] [Algebra R A'] (f : A →ₐ[R] A') :
    map f.toLinearMap (M * N) = map f.toLinearMap M * map f.toLinearMap N :=
  calc
    map f.toLinearMap (M * N) = ⨆ i : M, (N.map (LinearMap.mul R A i)).map f.toLinearMap := by
      /-
        R : Type u
        inst✝⁴ : CommSemiring R
        A : Type v
        inst✝³ : Semiring A
        inst✝² : Algebra R A
        M N : Submodule R A
        A' : Type u_1
        inst✝¹ : Semiring A'
        inst✝ : Algebra R A'
        f : AlgHom R A A'
        ⊢ Eq (Submodule.map f.toLinearMap (HMul.hMul M N)) (iSup fun i => Submodule.ma …
      -/
      rw [mul_eq_map₂]; apply map_iSup
                        /-
                          🎉 no goals
                        -/
    _ = map f.toLinearMap M * map f.toLinearMap N := by
      /-
        R : Type u
        inst✝⁴ : CommSemiring R
        A : Type v
        inst✝³ : Semiring A
        inst✝² : Algebra R A
        M N : Submodule R A
        A' : Type u_1
        inst✝¹ : Semiring A'
        inst✝ : Algebra R A'
        f : AlgHom R A A'
        ⊢ Eq (iSup fun i => Submodule.map f.toLinearMap (Submodule.map ((LinearMap.mul …
      -/
      rw [mul_eq_map₂]
      /-
        R : Type u
        inst✝⁴ : CommSemiring R
        A : Type v
        inst✝³ : Semiring A
        inst✝² : Algebra R A
        M N : Submodule R A
        A' : Type u_1
        inst✝¹ : Semiring A'
        inst✝ : Algebra R A'
        f : AlgHom R A A'
        ⊢ Eq (iSup fun i => Submodule.map f.toLinearMap (Submodule.map ((LinearMap.mul …
      -/
      apply congr_arg sSup
      /-
        R : Type u
        inst✝⁴ : CommSemiring R
        A : Type v
        inst✝³ : Semiring A
        inst✝² : Algebra R A
        M N : Submodule R A
        A' : Type u_1
        inst✝¹ : Semiring A'
        inst✝ : Algebra R A'
        f : AlgHom R A A'
        ⊢ Eq (Set.range fun i => Submodule.map f.toLinearMap (Submodule.map ((LinearMa …
      -/
      ext S
      /-
        case h
        R : Type u
        inst✝⁴ : CommSemiring R
        A : Type v
        inst✝³ : Semiring A
        inst✝² : Algebra R A
        M N : Submodule R A
        A' : Type u_1
        inst✝¹ : Semiring A'
        inst✝ : Algebra R A'
        f : AlgHom R A A'
        S : Submodule R A'
        ⊢ Iff (Membership.mem (Set.range fun i => Submodule.map f.toLinearMap (Submodu …
      -/
      constructor <;> rintro ⟨y, hy⟩
        /-
          case h.mp.intro
          R : Type u
          inst✝⁴ : CommSemiring R
          A : Type v
          inst✝³ : Semiring A
          inst✝² : Algebra R A
          M N : Submodule R A
          A' : Type u_1
          inst✝¹ : Semiring A'
          inst✝ : Algebra R A'
          f : AlgHom R A A'
          S : Submodule R A'
          y : Subtype fun x => Membership.mem M x
          hy : Eq ((fun i => Submodule.map f.toLinearMap (Submodule.map ((LinearMap.mul  …
          ⊢ Membership.mem (Set.range fun s => Submodule.map ((LinearMap.mul R A') ↑s) ( …
        -/
      · use ⟨f y, mem_map.mpr ⟨y.1, y.2, rfl⟩⟩  -- Porting note: added `⟨⟩`
        /-
          case h
          R : Type u
          inst✝⁴ : CommSemiring R
          A : Type v
          inst✝³ : Semiring A
          inst✝² : Algebra R A
          M N : Submodule R A
          A' : Type u_1
          inst✝¹ : Semiring A'
          inst✝ : Algebra R A'
          f : AlgHom R A A'
          S : Submodule R A'
          y : Subtype fun x => Membership.mem M x
          hy : Eq ((fun i => Submodule.map f.toLinearMap (Submodule.map ((LinearMap.mul  …
          ⊢ Eq ((fun s => Submodule.map ((LinearMap.mul R A') ↑s) (Submodule.map f.toLin …
        -/
        refine Eq.trans ?_ hy
        /-
          case h
          R : Type u
          inst✝⁴ : CommSemiring R
          A : Type v
          inst✝³ : Semiring A
          inst✝² : Algebra R A
          M N : Submodule R A
          A' : Type u_1
          inst✝¹ : Semiring A'
          inst✝ : Algebra R A'
          f : AlgHom R A A'
          S : Submodule R A'
          y : Subtype fun x => Membership.mem M x
          hy : Eq ((fun i => Submodule.map f.toLinearMap (Submodule.map ((LinearMap.mul  …
          ⊢ Eq ((fun s => Submodule.map ((LinearMap.mul R A') ↑s) (Submodule.map f.toLin …
        -/
        ext
        /-
          case h.h
          R : Type u
          inst✝⁴ : CommSemiring R
          A : Type v
          inst✝³ : Semiring A
          inst✝² : Algebra R A
          M N : Submodule R A
          A' : Type u_1
          inst✝¹ : Semiring A'
          inst✝ : Algebra R A'
          f : AlgHom R A A'
          S : Submodule R A'
          y : Subtype fun x => Membership.mem M x
          hy : Eq ((fun i => Submodule.map f.toLinearMap (Submodule.map ((LinearMap.mul  …
          x✝ : A'
          ⊢ Iff (Membership.mem ((fun s => Submodule.map ((LinearMap.mul R A') ↑s) (Subm …
        -/
        simp
        /-
          🎉 no goals
        -/
        /-
          case h.mpr.intro
          R : Type u
          inst✝⁴ : CommSemiring R
          A : Type v
          inst✝³ : Semiring A
          inst✝² : Algebra R A
          M N : Submodule R A
          A' : Type u_1
          inst✝¹ : Semiring A'
          inst✝ : Algebra R A'
          f : AlgHom R A A'
          S : Submodule R A'
          y : Subtype fun x => Membership.mem (Submodule.map f.toLinearMap M) x
          hy : Eq ((fun s => Submodule.map ((LinearMap.mul R A') ↑s) (Submodule.map f.to …
          ⊢ Membership.mem (Set.range fun i => Submodule.map f.toLinearMap (Submodule.ma …
        -/
      · obtain ⟨y', hy', fy_eq⟩ := mem_map.mp y.2
        /-
          case h.mpr.intro.intro.intro
          R : Type u
          inst✝⁴ : CommSemiring R
          A : Type v
          inst✝³ : Semiring A
          inst✝² : Algebra R A
          M N : Submodule R A
          A' : Type u_1
          inst✝¹ : Semiring A'
          inst✝ : Algebra R A'
          f : AlgHom R A A'
          S : Submodule R A'
          y : Subtype fun x => Membership.mem (Submodule.map f.toLinearMap M) x
          hy : Eq ((fun s => Submodule.map ((LinearMap.mul R A') ↑s) (Submodule.map f.to …
          y' : A
          hy' : Membership.mem M y'
          fy_eq : Eq (f.toLinearMap y') ↑y
          ⊢ Membership.mem (Set.range fun i => Submodule.map f.toLinearMap (Submodule.ma …
        -/
        use ⟨y', hy'⟩  -- Porting note: added `⟨⟩`
        /-
          case h
          R : Type u
          inst✝⁴ : CommSemiring R
          A : Type v
          inst✝³ : Semiring A
          inst✝² : Algebra R A
          M N : Submodule R A
          A' : Type u_1
          inst✝¹ : Semiring A'
          inst✝ : Algebra R A'
          f : AlgHom R A A'
          S : Submodule R A'
          y : Subtype fun x => Membership.mem (Submodule.map f.toLinearMap M) x
          hy : Eq ((fun s => Submodule.map ((LinearMap.mul R A') ↑s) (Submodule.map f.to …
          y' : A
          hy' : Membership.mem M y'
          fy_eq : Eq (f.toLinearMap y') ↑y
          ⊢ Eq ((fun i => Submodule.map f.toLinearMap (Submodule.map ((LinearMap.mul R A …
        -/
        refine Eq.trans ?_ hy
        /-
          case h
          R : Type u
          inst✝⁴ : CommSemiring R
          A : Type v
          inst✝³ : Semiring A
          inst✝² : Algebra R A
          M N : Submodule R A
          A' : Type u_1
          inst✝¹ : Semiring A'
          inst✝ : Algebra R A'
          f : AlgHom R A A'
          S : Submodule R A'
          y : Subtype fun x => Membership.mem (Submodule.map f.toLinearMap M) x
          hy : Eq ((fun s => Submodule.map ((LinearMap.mul R A') ↑s) (Submodule.map f.to …
          y' : A
          hy' : Membership.mem M y'
          fy_eq : Eq (f.toLinearMap y') ↑y
          ⊢ Eq ((fun i => Submodule.map f.toLinearMap (Submodule.map ((LinearMap.mul R A …
        -/
        rw [f.toLinearMap_apply] at fy_eq
        /-
          case h
          R : Type u
          inst✝⁴ : CommSemiring R
          A : Type v
          inst✝³ : Semiring A
          inst✝² : Algebra R A
          M N : Submodule R A
          A' : Type u_1
          inst✝¹ : Semiring A'
          inst✝ : Algebra R A'
          f : AlgHom R A A'
          S : Submodule R A'
          y : Subtype fun x => Membership.mem (Submodule.map f.toLinearMap M) x
          hy : Eq ((fun s => Submodule.map ((LinearMap.mul R A') ↑s) (Submodule.map f.to …
          y' : A
          hy' : Membership.mem M y'
          fy_eq : Eq (f y') ↑y
          ⊢ Eq ((fun i => Submodule.map f.toLinearMap (Submodule.map ((LinearMap.mul R A …
        -/
        ext
        /-
          case h.h
          R : Type u
          inst✝⁴ : CommSemiring R
          A : Type v
          inst✝³ : Semiring A
          inst✝² : Algebra R A
          M N : Submodule R A
          A' : Type u_1
          inst✝¹ : Semiring A'
          inst✝ : Algebra R A'
          f : AlgHom R A A'
          S : Submodule R A'
          y : Subtype fun x => Membership.mem (Submodule.map f.toLinearMap M) x
          hy : Eq ((fun s => Submodule.map ((LinearMap.mul R A') ↑s) (Submodule.map f.to …
          y' : A
          hy' : Membership.mem M y'
          fy_eq : Eq (f y') ↑y
          x✝ : A'
          ⊢ Iff (Membership.mem ((fun i => Submodule.map f.toLinearMap (Submodule.map (( …
        -/
        simp [fy_eq]
        /-
          🎉 no goals
        -/


theorem map_op_mul :
    map (↑(opLinearEquiv R : A ≃ₗ[R] Aᵐᵒᵖ) : A →ₗ[R] Aᵐᵒᵖ) (M * N) =
      map (↑(opLinearEquiv R : A ≃ₗ[R] Aᵐᵒᵖ) : A →ₗ[R] Aᵐᵒᵖ) N *
        map (↑(opLinearEquiv R : A ≃ₗ[R] Aᵐᵒᵖ) : A →ₗ[R] Aᵐᵒᵖ) M := by
  /-
    R : Type u
    inst✝² : CommSemiring R
    A : Type v
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    M N : Submodule R A
    ⊢ Eq (Submodule.map (↑(MulOpposite.opLinearEquiv R)) (HMul.hMul M N)) (HMul.hM …
  -/
  apply le_antisymm
    /-
      case a
      R : Type u
      inst✝² : CommSemiring R
      A : Type v
      inst✝¹ : Semiring A
      inst✝ : Algebra R A
      M N : Submodule R A
      ⊢ LE.le (Submodule.map (↑(MulOpposite.opLinearEquiv R)) (HMul.hMul M N)) (HMul …
    -/
  · simp_rw [map_le_iff_le_comap]
    /-
      case a
      R : Type u
      inst✝² : CommSemiring R
      A : Type v
      inst✝¹ : Semiring A
      inst✝ : Algebra R A
      M N : Submodule R A
      ⊢ LE.le (HMul.hMul M N) (Submodule.comap (↑(MulOpposite.opLinearEquiv R)) (HMu …
    -/
    refine mul_le.2 fun m hm n hn => ?_
    /-
      case a
      R : Type u
      inst✝² : CommSemiring R
      A : Type v
      inst✝¹ : Semiring A
      inst✝ : Algebra R A
      M N : Submodule R A
      m : A
      hm : Membership.mem M m
      n : A
      hn : Membership.mem N n
      ⊢ Membership.mem (Submodule.comap (↑(MulOpposite.opLinearEquiv R)) (HMul.hMul  …
    -/
    rw [mem_comap, map_equiv_eq_comap_symm, map_equiv_eq_comap_symm]
    /-
      case a
      R : Type u
      inst✝² : CommSemiring R
      A : Type v
      inst✝¹ : Semiring A
      inst✝ : Algebra R A
      M N : Submodule R A
      m : A
      hm : Membership.mem M m
      n : A
      hn : Membership.mem N n
      ⊢ Membership.mem (HMul.hMul (Submodule.comap (↑(MulOpposite.opLinearEquiv R).s …
    -/
    show op n * op m ∈ _
    /-
      case a
      R : Type u
      inst✝² : CommSemiring R
      A : Type v
      inst✝¹ : Semiring A
      inst✝ : Algebra R A
      M N : Submodule R A
      m : A
      hm : Membership.mem M m
      n : A
      hn : Membership.mem N n
      ⊢ Membership.mem (HMul.hMul (Submodule.comap (↑(MulOpposite.opLinearEquiv R).s …
    -/
    exact mul_mem_mul hn hm
    /-
      🎉 no goals
    -/
    /-
      case a
      R : Type u
      inst✝² : CommSemiring R
      A : Type v
      inst✝¹ : Semiring A
      inst✝ : Algebra R A
      M N : Submodule R A
      ⊢ LE.le (HMul.hMul (Submodule.map (↑(MulOpposite.opLinearEquiv R)) N) (Submodu …
    -/
  · refine mul_le.2 (MulOpposite.rec' fun m hm => MulOpposite.rec' fun n hn => ?_)
    /-
      case a
      R : Type u
      inst✝² : CommSemiring R
      A : Type v
      inst✝¹ : Semiring A
      inst✝ : Algebra R A
      M N : Submodule R A
      m : A
      hm : Membership.mem (Submodule.map (↑(MulOpposite.opLinearEquiv R)) N) (MulOpp …
      n : A
      hn : Membership.mem (Submodule.map (↑(MulOpposite.opLinearEquiv R)) M) (MulOpp …
      ⊢ Membership.mem (Submodule.map (↑(MulOpposite.opLinearEquiv R)) (HMul.hMul M  …
    -/
    rw [Submodule.mem_map_equiv] at hm hn ⊢
    /-
      case a
      R : Type u
      inst✝² : CommSemiring R
      A : Type v
      inst✝¹ : Semiring A
      inst✝ : Algebra R A
      M N : Submodule R A
      m : A
      hm : Membership.mem N ((MulOpposite.opLinearEquiv R).symm (MulOpposite.op m))
      n : A
      hn : Membership.mem M ((MulOpposite.opLinearEquiv R).symm (MulOpposite.op n))
      ⊢ Membership.mem (HMul.hMul M N) ((MulOpposite.opLinearEquiv R).symm (HMul.hMu …
    -/
    exact mul_mem_mul hn hm
    /-
      🎉 no goals
    -/


theorem comap_unop_mul :
    comap (↑(opLinearEquiv R : A ≃ₗ[R] Aᵐᵒᵖ).symm : Aᵐᵒᵖ →ₗ[R] A) (M * N) =
      comap (↑(opLinearEquiv R : A ≃ₗ[R] Aᵐᵒᵖ).symm : Aᵐᵒᵖ →ₗ[R] A) N *
        comap (↑(opLinearEquiv R : A ≃ₗ[R] Aᵐᵒᵖ).symm : Aᵐᵒᵖ →ₗ[R] A) M := by
  /-
    R : Type u
    inst✝² : CommSemiring R
    A : Type v
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    M N : Submodule R A
    ⊢ Eq (Submodule.comap (↑(MulOpposite.opLinearEquiv R).symm) (HMul.hMul M N)) ( …
  -/
  simp_rw [← map_equiv_eq_comap_symm, map_op_mul]
  /-
    🎉 no goals
  -/


theorem map_unop_mul (M N : Submodule R Aᵐᵒᵖ) :
    map (↑(opLinearEquiv R : A ≃ₗ[R] Aᵐᵒᵖ).symm : Aᵐᵒᵖ →ₗ[R] A) (M * N) =
      map (↑(opLinearEquiv R : A ≃ₗ[R] Aᵐᵒᵖ).symm : Aᵐᵒᵖ →ₗ[R] A) N *
        map (↑(opLinearEquiv R : A ≃ₗ[R] Aᵐᵒᵖ).symm : Aᵐᵒᵖ →ₗ[R] A) M :=
  have : Function.Injective (↑(opLinearEquiv R : A ≃ₗ[R] Aᵐᵒᵖ) : A →ₗ[R] Aᵐᵒᵖ) :=
    LinearEquiv.injective _
  map_injective_of_injective this <| by
    rw [← map_comp, map_op_mul, ← map_comp, ← map_comp, LinearEquiv.comp_coe,
      LinearEquiv.symm_trans_self, LinearEquiv.refl_toLinearMap, map_id, map_id, map_id]


theorem comap_op_mul (M N : Submodule R Aᵐᵒᵖ) :
    comap (↑(opLinearEquiv R : A ≃ₗ[R] Aᵐᵒᵖ) : A →ₗ[R] Aᵐᵒᵖ) (M * N) =
      comap (↑(opLinearEquiv R : A ≃ₗ[R] Aᵐᵒᵖ) : A →ₗ[R] Aᵐᵒᵖ) N *
        comap (↑(opLinearEquiv R : A ≃ₗ[R] Aᵐᵒᵖ) : A →ₗ[R] Aᵐᵒᵖ) M := by
  /-
    R : Type u
    inst✝² : CommSemiring R
    A : Type v
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    M N : Submodule R (MulOpposite A)
    ⊢ Eq (Submodule.comap (↑(MulOpposite.opLinearEquiv R)) (HMul.hMul M N)) (HMul. …
  -/
  simp_rw [comap_equiv_eq_map_symm, map_unop_mul]
  /-
    🎉 no goals
  -/


instance [IsScalarTower α A A] : IsScalarTower α (Submodule R A) (Submodule R A) where
  smul_assoc a S T := by
    rw [← S.span_eq, ← T.span_eq, smul_span, smul_eq_mul, smul_eq_mul, span_mul_span, span_mul_span,
      smul_span, smul_mul_assoc]


instance [SMulCommClass α A A] : SMulCommClass α (Submodule R A) (Submodule R A) where
  smul_comm a S T := by
    rw [← S.span_eq, ← T.span_eq, smul_span, smul_eq_mul, smul_eq_mul, span_mul_span, span_mul_span,
      smul_span, mul_smul_comm]


instance [SMulCommClass A α A] : SMulCommClass (Submodule R A) α (Submodule R A) :=
  have := SMulCommClass.symm A α A; .symm ..


/-- `Submodule.pointwiseNeg` distributes over multiplication.

This is available as an instance in the `Pointwise` locale. -/
protected def hasDistribPointwiseNeg {A} [Ring A] [Algebra R A] : HasDistribNeg (Submodule R A) :=
  toAddSubmonoid_injective.hasDistribNeg _ neg_toAddSubmonoid mul_toAddSubmonoid


theorem mem_span_mul_finite_of_mem_span_mul {R A} [Semiring R] [AddCommMonoid A] [Mul A]
    [Module R A] {S : Set A} {S' : Set A} {x : A} (hx : x ∈ span R (S * S')) :
    ∃ T T' : Finset A, ↑T ⊆ S ∧ ↑T' ⊆ S' ∧ x ∈ span R (T * T' : Set A) := by
  classical
  obtain ⟨U, h, hU⟩ := mem_span_finite_of_mem_span hx
  obtain ⟨T, T', hS, hS', h⟩ := Finset.subset_mul h
  use T, T', hS, hS'
  have h' : (U : Set A) ⊆ T * T' := by assumption_mod_cast
  have h'' := span_mono h' hU
  assumption


theorem mul_eq_span_mul_set (s t : Submodule R A) : s * t = span R ((s : Set A) * (t : Set A)) := by
  /-
    R : Type u
    inst✝² : CommSemiring R
    A : Type v
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    s t : Submodule R A
    ⊢ Eq (HMul.hMul s t) (Submodule.span R (HMul.hMul ↑s ↑t))
  -/
  rw [mul_eq_map₂]; exact map₂_eq_span_image2 _ s t
                    /-
                      🎉 no goals
                    -/


theorem mem_span_mul_finite_of_mem_mul {P Q : Submodule R A} {x : A} (hx : x ∈ P * Q) :
    ∃ T T' : Finset A, (T : Set A) ⊆ P ∧ (T' : Set A) ⊆ Q ∧ x ∈ span R (T * T' : Set A) :=
  Submodule.mem_span_mul_finite_of_mem_span_mul
        /-
          R : Type u
          inst✝² : CommSemiring R
          A : Type v
          inst✝¹ : Semiring A
          inst✝ : Algebra R A
          P Q : Submodule R A
          x : A
          hx : Membership.mem (HMul.hMul P Q) x
          ⊢ Membership.mem (Submodule.span R (HMul.hMul ↑P ↑Q)) x
        -/
    (by rwa [← Submodule.span_eq P, ← Submodule.span_eq Q, Submodule.span_mul_span] at hx)
        /-
          🎉 no goals
        -/


theorem mem_span_singleton_mul {x y : A} : x ∈ span R {y} * P ↔ ∃ z ∈ P, y * z = x := by
  /-
    R : Type u
    inst✝² : CommSemiring R
    A : Type v
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    P : Submodule R A
    x y : A
    ⊢ Iff (Membership.mem (HMul.hMul (Submodule.span R (Singleton.singleton y)) P) …
  -/
  simp_rw [mul_eq_map₂, (· * ·), map₂_span_singleton_eq_map]
  /-
    R : Type u
    inst✝² : CommSemiring R
    A : Type v
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    P : Submodule R A
    x y : A
    ⊢ Iff (Membership.mem (Submodule.map ((LinearMap.mul R A) y) P) x) (Exists fun …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem mem_mul_span_singleton {x y : A} : x ∈ P * span R {y} ↔ ∃ z ∈ P, z * y = x := by
  /-
    R : Type u
    inst✝² : CommSemiring R
    A : Type v
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    P : Submodule R A
    x y : A
    ⊢ Iff (Membership.mem (HMul.hMul P (Submodule.span R (Singleton.singleton y))) …
  -/
  simp_rw [mul_eq_map₂, (· * ·), map₂_span_singleton_eq_map_flip]
  /-
    R : Type u
    inst✝² : CommSemiring R
    A : Type v
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    P : Submodule R A
    x y : A
    ⊢ Iff (Membership.mem (Submodule.map ((LinearMap.mul R A).flip y) P) x) (Exist …
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma span_singleton_mul {x : A} {p : Submodule R A} :
    Submodule.span R {x} * p = x • p := ext fun _ ↦ mem_span_singleton_mul


lemma mem_smul_iff_inv_mul_mem {S} [Field S] [Algebra R S] {x : S} {p : Submodule R S} {y : S}
    (hx : x ≠ 0) : y ∈ x • p ↔ x⁻¹ * y ∈ p := by
  /-
    R : Type u
    inst✝² : CommSemiring R
    S : Type u_1
    inst✝¹ : Field S
    inst✝ : Algebra R S
    x : S
    p : Submodule R S
    y : S
    hx : Ne x 0
    ⊢ Iff (Membership.mem (HSMul.hSMul x p) y) (Membership.mem p (HMul.hMul (Inv.i …
  -/
  constructor
    /-
      case mp
      R : Type u
      inst✝² : CommSemiring R
      S : Type u_1
      inst✝¹ : Field S
      inst✝ : Algebra R S
      x : S
      p : Submodule R S
      y : S
      hx : Ne x 0
      ⊢ Membership.mem (HSMul.hSMul x p) y → Membership.mem p (HMul.hMul (Inv.inv x) …
    -/
  · rintro ⟨a, ha : a ∈ p, rfl⟩; simpa [inv_mul_cancel_left₀ hx]
                                 /-
                                   🎉 no goals
                                 -/
    /-
      case mpr
      R : Type u
      inst✝² : CommSemiring R
      S : Type u_1
      inst✝¹ : Field S
      inst✝ : Algebra R S
      x : S
      p : Submodule R S
      y : S
      hx : Ne x 0
      ⊢ Membership.mem p (HMul.hMul (Inv.inv x) y) → Membership.mem (HSMul.hSMul x p …
    -/
  · exact fun h ↦ ⟨_, h, by simp [mul_inv_cancel_left₀ hx]⟩
    /-
      🎉 no goals
    -/


lemma mul_mem_smul_iff {S} [CommRing S] [Algebra R S] {x : S} {p : Submodule R S} {y : S}
    (hx : x ∈ nonZeroDivisors S) :
    x * y ∈ x • p ↔ y ∈ p :=
                       /-
                         R : Type u
                         inst✝² : CommSemiring R
                         S : Type u_1
                         inst✝¹ : CommRing S
                         inst✝ : Algebra R S
                         x : S
                         p : Submodule R S
                         y : S
                         hx : Membership.mem (nonZeroDivisors S) x
                         ⊢ Iff (Exists fun a => And (Membership.mem (↑p) a) (Eq ((DistribMulAction.toLi …
                       -/
  show Exists _ ↔ _ by simp [mul_cancel_left_mem_nonZeroDivisors hx]
                       /-
                         🎉 no goals
                       -/


variable (M N) in
theorem mul_smul_mul_eq_smul_mul_smul (x y : R) : (x * y) • (M * N) = (x • M) * (y • N) := by
  /-
    R : Type u
    inst✝² : CommSemiring R
    A : Type v
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    M N : Submodule R A
    x y : R
    ⊢ Eq (HSMul.hSMul (HMul.hMul x y) (HMul.hMul M N)) (HMul.hMul (HSMul.hSMul x M …
  -/
  ext
  /-
    case h
    R : Type u
    inst✝² : CommSemiring R
    A : Type v
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    M N : Submodule R A
    x y : R
    x✝ : A
    ⊢ Iff (Membership.mem (HSMul.hSMul (HMul.hMul x y) (HMul.hMul M N)) x✝) (Membe …
  -/
  refine ⟨?_, fun hx ↦ Submodule.mul_induction_on hx ?_ fun _ _ hx hy ↦ Submodule.add_mem _ hx hy⟩
    /-
      case h.refine_1
      R : Type u
      inst✝² : CommSemiring R
      A : Type v
      inst✝¹ : Semiring A
      inst✝ : Algebra R A
      M N : Submodule R A
      x y : R
      x✝ : A
      ⊢ Membership.mem (HSMul.hSMul (HMul.hMul x y) (HMul.hMul M N)) x✝ → Membership …
    -/
  · rintro ⟨_, hx, rfl⟩
    /-
      case h.refine_1.intro.intro
      R : Type u
      inst✝² : CommSemiring R
      A : Type v
      inst✝¹ : Semiring A
      inst✝ : Algebra R A
      M N : Submodule R A
      x y : R
      w✝ : A
      hx : Membership.mem (↑(HMul.hMul M N)) w✝
      ⊢ Membership.mem (HMul.hMul (HSMul.hSMul x M) (HSMul.hSMul y N)) ((DistribMulA …
    -/
    rw [DistribMulAction.toLinearMap_apply]
    /-
      case h.refine_1.intro.intro
      R : Type u
      inst✝² : CommSemiring R
      A : Type v
      inst✝¹ : Semiring A
      inst✝ : Algebra R A
      M N : Submodule R A
      x y : R
      w✝ : A
      hx : Membership.mem (↑(HMul.hMul M N)) w✝
      ⊢ Membership.mem (HMul.hMul (HSMul.hSMul x M) (HSMul.hSMul y N)) (HSMul.hSMul  …
    -/
    refine Submodule.mul_induction_on hx (fun m hm n hn ↦ ?_) (fun _ _ hn hm ↦ ?_)
      /-
        case h.refine_1.intro.intro.refine_1
        R : Type u
        inst✝² : CommSemiring R
        A : Type v
        inst✝¹ : Semiring A
        inst✝ : Algebra R A
        M N : Submodule R A
        x y : R
        w✝ : A
        hx : Membership.mem (↑(HMul.hMul M N)) w✝
        m : A
        hm : Membership.mem M m
        n : A
        hn : Membership.mem N n
        ⊢ Membership.mem (HMul.hMul (HSMul.hSMul x M) (HSMul.hSMul y N)) (HSMul.hSMul  …
      -/
    · rw [mul_smul_mul_comm]
      /-
        case h.refine_1.intro.intro.refine_1
        R : Type u
        inst✝² : CommSemiring R
        A : Type v
        inst✝¹ : Semiring A
        inst✝ : Algebra R A
        M N : Submodule R A
        x y : R
        w✝ : A
        hx : Membership.mem (↑(HMul.hMul M N)) w✝
        m : A
        hm : Membership.mem M m
        n : A
        hn : Membership.mem N n
        ⊢ Membership.mem (HMul.hMul (HSMul.hSMul x M) (HSMul.hSMul y N)) (HMul.hMul (H …
      -/
      exact mul_mem_mul (smul_mem_pointwise_smul m x M hm) (smul_mem_pointwise_smul n y N hn)
      /-
        🎉 no goals
      -/
      /-
        case h.refine_1.intro.intro.refine_2
        R : Type u
        inst✝² : CommSemiring R
        A : Type v
        inst✝¹ : Semiring A
        inst✝ : Algebra R A
        M N : Submodule R A
        x y : R
        w✝ : A
        hx : Membership.mem (↑(HMul.hMul M N)) w✝
        x✝¹ x✝ : A
        hn : Membership.mem (HMul.hMul (HSMul.hSMul x M) (HSMul.hSMul y N)) (HSMul.hSM …
        hm : Membership.mem (HMul.hMul (HSMul.hSMul x M) (HSMul.hSMul y N)) (HSMul.hSM …
        ⊢ Membership.mem (HMul.hMul (HSMul.hSMul x M) (HSMul.hSMul y N)) (HSMul.hSMul  …
      -/
    · rw [smul_add]
      /-
        case h.refine_1.intro.intro.refine_2
        R : Type u
        inst✝² : CommSemiring R
        A : Type v
        inst✝¹ : Semiring A
        inst✝ : Algebra R A
        M N : Submodule R A
        x y : R
        w✝ : A
        hx : Membership.mem (↑(HMul.hMul M N)) w✝
        x✝¹ x✝ : A
        hn : Membership.mem (HMul.hMul (HSMul.hSMul x M) (HSMul.hSMul y N)) (HSMul.hSM …
        hm : Membership.mem (HMul.hMul (HSMul.hSMul x M) (HSMul.hSMul y N)) (HSMul.hSM …
        ⊢ Membership.mem (HMul.hMul (HSMul.hSMul x M) (HSMul.hSMul y N)) (HAdd.hAdd (H …
      -/
      exact Submodule.add_mem _ hn hm
      /-
        🎉 no goals
      -/
    /-
      case h.refine_2
      R : Type u
      inst✝² : CommSemiring R
      A : Type v
      inst✝¹ : Semiring A
      inst✝ : Algebra R A
      M N : Submodule R A
      x y : R
      x✝ : A
      hx : Membership.mem (HMul.hMul (HSMul.hSMul x M) (HSMul.hSMul y N)) x✝
      ⊢ ∀ (m : A), Membership.mem (HSMul.hSMul x M) m → ∀ (n : A), Membership.mem (H …
    -/
  · rintro _ ⟨m, hm, rfl⟩ _ ⟨n, hn, rfl⟩
    /-
      case h.refine_2.intro.intro.intro.intro
      R : Type u
      inst✝² : CommSemiring R
      A : Type v
      inst✝¹ : Semiring A
      inst✝ : Algebra R A
      M N : Submodule R A
      x y : R
      x✝ : A
      hx : Membership.mem (HMul.hMul (HSMul.hSMul x M) (HSMul.hSMul y N)) x✝
      m : A
      hm : Membership.mem (↑M) m
      n : A
      hn : Membership.mem (↑N) n
      ⊢ Membership.mem (HSMul.hSMul (HMul.hMul x y) (HMul.hMul M N)) (HMul.hMul ((Di …
    -/
    simp_rw [DistribMulAction.toLinearMap_apply, smul_mul_smul_comm]
    /-
      case h.refine_2.intro.intro.intro.intro
      R : Type u
      inst✝² : CommSemiring R
      A : Type v
      inst✝¹ : Semiring A
      inst✝ : Algebra R A
      M N : Submodule R A
      x y : R
      x✝ : A
      hx : Membership.mem (HMul.hMul (HSMul.hSMul x M) (HSMul.hSMul y N)) x✝
      m : A
      hm : Membership.mem (↑M) m
      n : A
      hn : Membership.mem (↑N) n
      ⊢ Membership.mem (HSMul.hSMul (HMul.hMul x y) (HMul.hMul M N)) (HSMul.hSMul (H …
    -/
    exact smul_mem_pointwise_smul _ _ _ (mul_mem_mul hm hn)
    /-
      🎉 no goals
    -/


/-- Sub-R-modules of an R-algebra form an idempotent semiring. -/
instance idemSemiring : IdemSemiring (Submodule R A) where
  __ := instNonUnitalSemiring
  one_mul := Submodule.one_mul
  mul_one := Submodule.mul_one
  bot_le _ := bot_le


theorem span_pow (s : Set A) : ∀ n : ℕ, span R s ^ n = span R (s ^ n)
            /-
              R : Type u
              inst✝² : CommSemiring R
              A : Type v
              inst✝¹ : Semiring A
              inst✝ : Algebra R A
              s : Set A
              ⊢ Eq (HPow.hPow (Submodule.span R s) 0) (Submodule.span R (HPow.hPow s 0))
            -/
  | 0 => by rw [pow_zero, pow_zero, one_eq_span_one_set]
            /-
              🎉 no goals
            -/
                /-
                  R : Type u
                  inst✝² : CommSemiring R
                  A : Type v
                  inst✝¹ : Semiring A
                  inst✝ : Algebra R A
                  s : Set A
                  n : Nat
                  ⊢ Eq (HPow.hPow (Submodule.span R s) (HAdd.hAdd n 1)) (Submodule.span R (HPow. …
                -/
  | n + 1 => by rw [pow_succ, pow_succ, span_pow s n, span_mul_span]
                /-
                  🎉 no goals
                -/


theorem pow_eq_span_pow_set (n : ℕ) : M ^ n = span R ((M : Set A) ^ n) := by
  /-
    R : Type u
    inst✝² : CommSemiring R
    A : Type v
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    M : Submodule R A
    n : Nat
    ⊢ Eq (HPow.hPow M n) (Submodule.span R (HPow.hPow (↑M) n))
  -/
  rw [← span_pow, span_eq]
  /-
    🎉 no goals
  -/


/-- Dependent version of `Submodule.pow_induction_on_left`. -/
@[elab_as_elim]
protected theorem pow_induction_on_left' {C : ∀ (n : ℕ) (x), x ∈ M ^ n → Prop}
    (algebraMap : ∀ r : R, C 0 (algebraMap _ _ r) (algebraMap_mem r))
    (add : ∀ x y i hx hy, C i x hx → C i y hy → C i (x + y) (add_mem ‹_› ‹_›))
    (mem_mul : ∀ m (hm : m ∈ M), ∀ (i x hx), C i x hx → C i.succ (m * x)
      ((pow_succ' M i).symm ▸ (mul_mem_mul hm hx)))
    -- Porting note: swapped argument order to match order of `C`
    {n : ℕ} {x : A}
    (hx : x ∈ M ^ n) : C n x hx := by
  induction n generalizing x with
  | zero =>
    rw [pow_zero] at hx
    obtain ⟨r, rfl⟩ := mem_one.mp hx
    exact algebraMap r
  | succ n n_ih =>
    revert hx
    simp_rw [pow_succ']
    exact fun hx ↦ Submodule.mul_induction_on' (fun m hm x ih => mem_mul _ hm _ _ _ (n_ih ih))
      (fun x hx y hy Cx Cy => add _ _ _ _ _ Cx Cy) hx


/-- Dependent version of `Submodule.pow_induction_on_right`. -/
@[elab_as_elim]
protected theorem pow_induction_on_right' {C : ∀ (n : ℕ) (x), x ∈ M ^ n → Prop}
    (algebraMap : ∀ r : R, C 0 (algebraMap _ _ r) (algebraMap_mem r))
    (add : ∀ x y i hx hy, C i x hx → C i y hy → C i (x + y) (add_mem ‹_› ‹_›))
    (mul_mem :
      ∀ i x hx, C i x hx →
        ∀ m (hm : m ∈ M), C i.succ (x * m) (mul_mem_mul hx hm))
    -- Porting note: swapped argument order to match order of `C`
    {n : ℕ} {x : A} (hx : x ∈ M ^ n) : C n x hx := by
  induction n generalizing x with
  | zero =>
    rw [pow_zero] at hx
    obtain ⟨r, rfl⟩ := mem_one.mp hx
    exact algebraMap r
  | succ n n_ih =>
    revert hx
    simp_rw [pow_succ]
    exact fun hx ↦ Submodule.mul_induction_on' (fun m hm x ih => mul_mem _ _ hm (n_ih _) _ ih)
      (fun x hx y hy Cx Cy => add _ _ _ _ _ Cx Cy) hx


/-- To show a property on elements of `M ^ n` holds, it suffices to show that it holds for scalars,
is closed under addition, and holds for `m * x` where `m ∈ M` and it holds for `x` -/
@[elab_as_elim]
protected theorem pow_induction_on_left {C : A → Prop} (hr : ∀ r : R, C (algebraMap _ _ r))
    (hadd : ∀ x y, C x → C y → C (x + y)) (hmul : ∀ m ∈ M, ∀ (x), C x → C (m * x)) {x : A} {n : ℕ}
    (hx : x ∈ M ^ n) : C x :=
  -- Porting note: `M` is explicit yet can't be passed positionally!
  Submodule.pow_induction_on_left' (M := M) (C := fun _ a _ => C a) hr
    (fun x y _i _hx _hy => hadd x y)
    (fun _m hm _i _x _hx => hmul _ hm _) hx


/-- To show a property on elements of `M ^ n` holds, it suffices to show that it holds for scalars,
is closed under addition, and holds for `x * m` where `m ∈ M` and it holds for `x` -/
@[elab_as_elim]
protected theorem pow_induction_on_right {C : A → Prop} (hr : ∀ r : R, C (algebraMap _ _ r))
    (hadd : ∀ x y, C x → C y → C (x + y)) (hmul : ∀ x, C x → ∀ m ∈ M, C (x * m)) {x : A} {n : ℕ}
    (hx : x ∈ M ^ n) : C x :=
  Submodule.pow_induction_on_right' (M := M) (C := fun _ a _ => C a) hr
    (fun x y _i _hx _hy => hadd x y)
    (fun _i _x _hx => hmul _) hx


/-- `Submonoid.map` as a `RingHom`, when applied to an `AlgHom`. -/
@[simps]
def mapHom {A'} [Semiring A'] [Algebra R A'] (f : A →ₐ[R] A') :
    Submodule R A →+* Submodule R A' where
  toFun := map f.toLinearMap
  map_zero' := Submodule.map_bot _
  map_add' := (Submodule.map_sup · · _)
  map_one' := Submodule.map_one _
  map_mul' := (Submodule.map_mul · · _)


theorem mapHom_id : mapHom (.id R A) = .id _ := RingHom.ext map_id


/-- The ring of submodules of the opposite algebra is isomorphic to the opposite ring of
submodules. -/
@[simps apply symm_apply]
def equivOpposite : Submodule R Aᵐᵒᵖ ≃+* (Submodule R A)ᵐᵒᵖ where
  toFun p := op <| p.comap (↑(opLinearEquiv R : A ≃ₗ[R] Aᵐᵒᵖ) : A →ₗ[R] Aᵐᵒᵖ)
  invFun p := p.unop.comap (↑(opLinearEquiv R : A ≃ₗ[R] Aᵐᵒᵖ).symm : Aᵐᵒᵖ →ₗ[R] A)
  left_inv _ := SetLike.coe_injective <| rfl
  right_inv _ := unop_injective <| SetLike.coe_injective rfl
                     /-
                       ι : Sort uι
                       R : Type u
                       inst✝² : CommSemiring R
                       A : Type v
                       inst✝¹ : Semiring A
                       inst✝ : Algebra R A
                       S T : Set A
                       M N P Q : Submodule R A
                       m n : A
                       p q : Submodule R (MulOpposite A)
                       ⊢ Eq ({ toFun := fun p => MulOpposite.op (Submodule.comap (↑(MulOpposite.opLin …
                     -/
  map_add' p q := by simp [comap_equiv_eq_map_symm, ← op_add]
                     /-
                       🎉 no goals
                     -/
  map_mul' _ _ := congr_arg op <| comap_op_mul _ _


protected theorem map_pow {A'} [Semiring A'] [Algebra R A'] (f : A →ₐ[R] A') (n : ℕ) :
    map f.toLinearMap (M ^ n) = map f.toLinearMap M ^ n :=
  map_pow (mapHom f) M n


theorem comap_unop_pow (n : ℕ) :
    comap (↑(opLinearEquiv R : A ≃ₗ[R] Aᵐᵒᵖ).symm : Aᵐᵒᵖ →ₗ[R] A) (M ^ n) =
      comap (↑(opLinearEquiv R : A ≃ₗ[R] Aᵐᵒᵖ).symm : Aᵐᵒᵖ →ₗ[R] A) M ^ n :=
  (equivOpposite : Submodule R Aᵐᵒᵖ ≃+* _).symm.map_pow (op M) n


theorem comap_op_pow (n : ℕ) (M : Submodule R Aᵐᵒᵖ) :
    comap (↑(opLinearEquiv R : A ≃ₗ[R] Aᵐᵒᵖ) : A →ₗ[R] Aᵐᵒᵖ) (M ^ n) =
      comap (↑(opLinearEquiv R : A ≃ₗ[R] Aᵐᵒᵖ) : A →ₗ[R] Aᵐᵒᵖ) M ^ n :=
  op_injective <| (equivOpposite : Submodule R Aᵐᵒᵖ ≃+* _).map_pow M n


theorem map_op_pow (n : ℕ) :
    map (↑(opLinearEquiv R : A ≃ₗ[R] Aᵐᵒᵖ) : A →ₗ[R] Aᵐᵒᵖ) (M ^ n) =
      map (↑(opLinearEquiv R : A ≃ₗ[R] Aᵐᵒᵖ) : A →ₗ[R] Aᵐᵒᵖ) M ^ n := by
  /-
    R : Type u
    inst✝² : CommSemiring R
    A : Type v
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    M : Submodule R A
    n : Nat
    ⊢ Eq (Submodule.map (↑(MulOpposite.opLinearEquiv R)) (HPow.hPow M n)) (HPow.hP …
  -/
  rw [map_equiv_eq_comap_symm, map_equiv_eq_comap_symm, comap_unop_pow]
  /-
    🎉 no goals
  -/


theorem map_unop_pow (n : ℕ) (M : Submodule R Aᵐᵒᵖ) :
    map (↑(opLinearEquiv R : A ≃ₗ[R] Aᵐᵒᵖ).symm : Aᵐᵒᵖ →ₗ[R] A) (M ^ n) =
      map (↑(opLinearEquiv R : A ≃ₗ[R] Aᵐᵒᵖ).symm : Aᵐᵒᵖ →ₗ[R] A) M ^ n := by
  /-
    R : Type u
    inst✝² : CommSemiring R
    A : Type v
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    n : Nat
    M : Submodule R (MulOpposite A)
    ⊢ Eq (Submodule.map (↑(MulOpposite.opLinearEquiv R).symm) (HPow.hPow M n)) (HP …
  -/
  rw [← comap_equiv_eq_map_symm, ← comap_equiv_eq_map_symm, comap_op_pow]
  /-
    🎉 no goals
  -/


/-- `span` is a semiring homomorphism (recall multiplication is pointwise multiplication of subsets
on either side). -/
@[simps]
def span.ringHom : SetSemiring A →+* Submodule R A where
  toFun s := Submodule.span R (SetSemiring.down s)
  map_zero' := span_empty
  map_one' := one_eq_span.symm
  map_add' := span_union
                     /-
                       ι : Sort uι
                       R : Type u
                       inst✝² : CommSemiring R
                       A : Type v
                       inst✝¹ : Semiring A
                       inst✝ : Algebra R A
                       S T : Set A
                       M N P Q : Submodule R A
                       m n : A
                       s t : SetSemiring A
                       ⊢ Eq ({ toFun := fun s => Submodule.span R (SetSemiring.down s), map_one' := ⋯ …
                     -/
  map_mul' s t := by simp_rw [SetSemiring.down_mul, span_mul_span]
                     /-
                       🎉 no goals
                     -/


/-- The action on a submodule corresponding to applying the action to every element.

This is available as an instance in the `Pointwise` locale.

This is a stronger version of `Submodule.pointwiseDistribMulAction`. -/
protected def pointwiseMulSemiringAction : MulSemiringAction α (Submodule R A) where
  __ := Submodule.pointwiseDistribMulAction
  smul_mul r x y := Submodule.map_mul x y <| MulSemiringAction.toAlgHom R A r
  smul_one r := Submodule.map_one <| MulSemiringAction.toAlgHom R A r


theorem mul_mem_mul_rev (hm : m ∈ M) (hn : n ∈ N) : n * m ∈ M * N :=
  mul_comm m n ▸ mul_mem_mul hm hn


protected theorem mul_comm : M * N = N * M :=
  le_antisymm (mul_le.2 fun _r hrm _s hsn => mul_mem_mul_rev hsn hrm)
    (mul_le.2 fun _r hrn _s hsm => mul_mem_mul_rev hsm hrn)


/-- Sub-R-modules of an R-algebra A form a semiring. -/
instance : IdemCommSemiring (Submodule R A) :=
  { Submodule.idemSemiring with mul_comm := Submodule.mul_comm }


theorem prod_span {ι : Type*} (s : Finset ι) (M : ι → Set A) :
    (∏ i ∈ s, Submodule.span R (M i)) = Submodule.span R (∏ i ∈ s, M i) := by
  /-
    R : Type u
    inst✝² : CommSemiring R
    A : Type v
    inst✝¹ : CommSemiring A
    inst✝ : Algebra R A
    ι : Type u_1
    s : Finset ι
    M : ι → Set A
    ⊢ Eq (s.prod fun i => Submodule.span R (M i)) (Submodule.span R (s.prod fun i  …
  -/
  letI := Classical.decEq ι
  /-
    R : Type u
    inst✝² : CommSemiring R
    A : Type v
    inst✝¹ : CommSemiring A
    inst✝ : Algebra R A
    ι : Type u_1
    s : Finset ι
    M : ι → Set A
    this : DecidableEq ι := Classical.decEq ι
    ⊢ Eq (s.prod fun i => Submodule.span R (M i)) (Submodule.span R (s.prod fun i  …
  -/
  refine Finset.induction_on s ?_ ?_
    /-
      case refine_1
      R : Type u
      inst✝² : CommSemiring R
      A : Type v
      inst✝¹ : CommSemiring A
      inst✝ : Algebra R A
      ι : Type u_1
      s : Finset ι
      M : ι → Set A
      this : DecidableEq ι := Classical.decEq ι
      ⊢ Eq (EmptyCollection.emptyCollection.prod fun i => Submodule.span R (M i)) (S …
    -/
  · simp [one_eq_span, Set.singleton_one]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      R : Type u
      inst✝² : CommSemiring R
      A : Type v
      inst✝¹ : CommSemiring A
      inst✝ : Algebra R A
      ι : Type u_1
      s : Finset ι
      M : ι → Set A
      this : DecidableEq ι := Classical.decEq ι
      ⊢ ∀ ⦃a : ι⦄ {s : Finset ι}, Not (Membership.mem s a) → Eq (s.prod fun i => Sub …
    -/
  · intro _ _ H ih
    /-
      case refine_2
      R : Type u
      inst✝² : CommSemiring R
      A : Type v
      inst✝¹ : CommSemiring A
      inst✝ : Algebra R A
      ι : Type u_1
      s : Finset ι
      M : ι → Set A
      this : DecidableEq ι := Classical.decEq ι
      a✝ : ι
      s✝ : Finset ι
      H : Not (Membership.mem s✝ a✝)
      ih : Eq (s✝.prod fun i => Submodule.span R (M i)) (Submodule.span R (s✝.prod f …
      ⊢ Eq ((Insert.insert a✝ s✝).prod fun i => Submodule.span R (M i)) (Submodule.s …
    -/
    rw [Finset.prod_insert H, Finset.prod_insert H, ih, span_mul_span]
    /-
      🎉 no goals
    -/


theorem prod_span_singleton {ι : Type*} (s : Finset ι) (x : ι → A) :
    (∏ i ∈ s, span R ({x i} : Set A)) = span R {∏ i ∈ s, x i} := by
  /-
    R : Type u
    inst✝² : CommSemiring R
    A : Type v
    inst✝¹ : CommSemiring A
    inst✝ : Algebra R A
    ι : Type u_1
    s : Finset ι
    x : ι → A
    ⊢ Eq (s.prod fun i => Submodule.span R (Singleton.singleton (x i))) (Submodule …
  -/
  rw [prod_span, Set.finset_prod_singleton]
  /-
    🎉 no goals
  -/


/-- R-submodules of the R-algebra A are a module over `Set A`. -/
instance moduleSet : Module (SetSemiring A) (Submodule R A) where
  -- Porting note: have to unfold both `HSMul.hSMul` and `SMul.smul`
  -- Note: the hint `(α := A)` is new in https://github.com/leanprover-community/mathlib4/pull/8386
  smul s P := span R (SetSemiring.down (α := A) s) * P
  smul_add _ _ _ := mul_add _ _ _
  add_smul s t P := by
    /-
      ι : Sort uι
      R : Type u
      inst✝² : CommSemiring R
      A : Type v
      inst✝¹ : CommSemiring A
      inst✝ : Algebra R A
      M N : Submodule R A
      m n : A
      s t : SetSemiring A
      P : Submodule R A
      ⊢ Eq (HSMul.hSMul (HAdd.hAdd s t) P) (HAdd.hAdd (HSMul.hSMul s P) (HSMul.hSMul …
    -/
    simp_rw [HSMul.hSMul, SetSemiring.down_add, span_union, sup_mul, add_eq_sup]
    /-
      ι : Sort uι
      R : Type u
      inst✝² : CommSemiring R
      A : Type v
      inst✝¹ : CommSemiring A
      inst✝ : Algebra R A
      M N : Submodule R A
      m n : A
      s t : SetSemiring A
      P : Submodule R A
      ⊢ Eq (HSMul.hSMul (HMul.hMul s t) P) (HSMul.hSMul s (HSMul.hSMul t P))
    -/
    /-
      🎉 no goals
    -/
    /-
      ι : Sort uι
      R : Type u
      inst✝² : CommSemiring R
      A : Type v
      inst✝¹ : CommSemiring A
      inst✝ : Algebra R A
      M N : Submodule R A
      m n : A
      P : Submodule R A
      ⊢ Eq (HSMul.hSMul 1 P) P
    -/
    /-
      🎉 no goals
    -/
    /-
      🎉 no goals
    -/
  mul_smul s t P := by
    simp_rw [HSMul.hSMul, SetSemiring.down_mul, ← mul_assoc, span_mul_span]
  one_smul P := by
    simp_rw [HSMul.hSMul, SetSemiring.down_one, ← one_eq_span_one_set, one_mul]
  zero_smul P := by
    /-
      ι : Sort uι
      R : Type u
      inst✝² : CommSemiring R
      A : Type v
      inst✝¹ : CommSemiring A
      inst✝ : Algebra R A
      M N : Submodule R A
      m n : A
      P : Submodule R A
      ⊢ Eq (HSMul.hSMul 0 P) 0
    -/
    simp_rw [HSMul.hSMul, SetSemiring.down_zero, span_empty, bot_mul, bot_eq_zero]
    /-
      🎉 no goals
    -/
  smul_zero _ := mul_bot _


theorem setSemiring_smul_def (s : SetSemiring A) (P : Submodule R A) :
    s • P = span R (SetSemiring.down (α := A) s) * P :=
  rfl


theorem smul_le_smul {s t : SetSemiring A} {M N : Submodule R A}
    (h₁ : SetSemiring.down (α := A) s ⊆ SetSemiring.down (α := A) t)
    (h₂ : M ≤ N) : s • M ≤ t • N :=
  mul_le_mul (span_mono h₁) h₂


theorem singleton_smul (a : A) (M : Submodule R A) :
    Set.up ({a} : Set A) • M = M.map (LinearMap.mulLeft R a) := by
  /-
    R : Type u
    inst✝² : CommSemiring R
    A : Type v
    inst✝¹ : CommSemiring A
    inst✝ : Algebra R A
    a : A
    M : Submodule R A
    ⊢ Eq (HSMul.hSMul (Set.up (Singleton.singleton a)) M) (Submodule.map (LinearMa …
  -/
  conv_lhs => rw [← span_eq M]
  /-
    R : Type u
    inst✝² : CommSemiring R
    A : Type v
    inst✝¹ : CommSemiring A
    inst✝ : Algebra R A
    a : A
    M : Submodule R A
    ⊢ Eq (HSMul.hSMul (Set.up (Singleton.singleton a)) (Submodule.span R ↑M)) (Sub …
  -/
  rw [setSemiring_smul_def, SetSemiring.down_up, span_mul_span, singleton_mul]
  /-
    R : Type u
    inst✝² : CommSemiring R
    A : Type v
    inst✝¹ : CommSemiring A
    inst✝ : Algebra R A
    a : A
    M : Submodule R A
    ⊢ Eq (Submodule.span R (Set.image (fun x => HMul.hMul a x) ↑M)) (Submodule.map …
  -/
  exact (map (LinearMap.mulLeft R a) M).span_eq
  /-
    🎉 no goals
  -/


/-- The elements of `I / J` are the `x` such that `x • J ⊆ I`.

In fact, we define `x ∈ I / J` to be `∀ y ∈ J, x * y ∈ I` (see `mem_div_iff_forall_mul_mem`),
which is equivalent to `x • J ⊆ I` (see `mem_div_iff_smul_subset`), but nicer to use in proofs.

This is the general form of the ideal quotient, traditionally written $I : J$.
-/
instance : Div (Submodule R A) :=
  ⟨fun I J =>
    { carrier := { x | ∀ y ∈ J, x * y ∈ I }
      zero_mem' := fun y _ => by
        /-
          ι : Sort uι
          R : Type u
          inst✝² : CommSemiring R
          A : Type v
          inst✝¹ : CommSemiring A
          inst✝ : Algebra R A
          M N : Submodule R A
          m n : A
          I J : Submodule R A
          y : A
          x✝ : Membership.mem J y
          ⊢ Membership.mem I (HMul.hMul 0 y)
        -/
        rw [zero_mul]
        /-
          ι : Sort uι
          R : Type u
          inst✝² : CommSemiring R
          A : Type v
          inst✝¹ : CommSemiring A
          inst✝ : Algebra R A
          M N : Submodule R A
          m n : A
          I J : Submodule R A
          y : A
          x✝ : Membership.mem J y
          ⊢ Membership.mem I 0
        -/
        /-
          ι : Sort uι
          R : Type u
          inst✝² : CommSemiring R
          A : Type v
          inst✝¹ : CommSemiring A
          inst✝ : Algebra R A
          M N : Submodule R A
          m n : A
          I J : Submodule R A
          a✝ b✝ : A
          ha : Membership.mem (setOf fun x => ∀ (y : A), Membership.mem J y → Membership …
          hb : Membership.mem (setOf fun x => ∀ (y : A), Membership.mem J y → Membership …
          y : A
          hy : Membership.mem J y
          ⊢ Membership.mem I (HMul.hMul (HAdd.hAdd a✝ b✝) y)
        -/
        apply Submodule.zero_mem
        /-
          ι : Sort uι
          R : Type u
          inst✝² : CommSemiring R
          A : Type v
          inst✝¹ : CommSemiring A
          inst✝ : Algebra R A
          M N : Submodule R A
          m n : A
          I J : Submodule R A
          a✝ b✝ : A
          ha : Membership.mem (setOf fun x => ∀ (y : A), Membership.mem J y → Membership …
          hb : Membership.mem (setOf fun x => ∀ (y : A), Membership.mem J y → Membership …
          y : A
          hy : Membership.mem J y
          ⊢ Membership.mem I (HAdd.hAdd (HMul.hMul a✝ y) (HMul.hMul b✝ y))
        -/
        /-
          🎉 no goals
        -/
        /-
          🎉 no goals
        -/
      add_mem' := fun ha hb y hy => by
        rw [add_mul]
        exact Submodule.add_mem _ (ha _ hy) (hb _ hy)
      smul_mem' := fun r x hx y hy => by
        /-
          ι : Sort uι
          R : Type u
          inst✝² : CommSemiring R
          A : Type v
          inst✝¹ : CommSemiring A
          inst✝ : Algebra R A
          M N : Submodule R A
          m n : A
          I J : Submodule R A
          r : R
          x : A
          hx : Membership.mem { carrier := setOf fun x => ∀ (y : A), Membership.mem J y  …
          y : A
          hy : Membership.mem J y
          ⊢ Membership.mem I (HMul.hMul (HSMul.hSMul r x) y)
        -/
        rw [Algebra.smul_mul_assoc]
        /-
          ι : Sort uι
          R : Type u
          inst✝² : CommSemiring R
          A : Type v
          inst✝¹ : CommSemiring A
          inst✝ : Algebra R A
          M N : Submodule R A
          m n : A
          I J : Submodule R A
          r : R
          x : A
          hx : Membership.mem { carrier := setOf fun x => ∀ (y : A), Membership.mem J y  …
          y : A
          hy : Membership.mem J y
          ⊢ Membership.mem I (HSMul.hSMul r (HMul.hMul x y))
        -/
        exact Submodule.smul_mem _ _ (hx _ hy) }⟩
        /-
          🎉 no goals
        -/


theorem mem_div_iff_forall_mul_mem {x : A} {I J : Submodule R A} : x ∈ I / J ↔ ∀ y ∈ J, x * y ∈ I :=
  Iff.refl _


theorem mem_div_iff_smul_subset {x : A} {I J : Submodule R A} : x ∈ I / J ↔ x • (J : Set A) ⊆ I :=
  ⟨fun h y ⟨y', hy', xy'_eq_y⟩ => by
    /-
      R : Type u
      inst✝² : CommSemiring R
      A : Type v
      inst✝¹ : CommSemiring A
      inst✝ : Algebra R A
      x : A
      I J : Submodule R A
      h : Membership.mem (HDiv.hDiv I J) x
      y : A
      x✝ : Membership.mem (HSMul.hSMul x ↑J) y
      y' : A
      hy' : Membership.mem (↑J) y'
      xy'_eq_y : Eq ((fun x_1 => HSMul.hSMul x x_1) y') y
      ⊢ Membership.mem (↑I) y
    -/
    rw [← xy'_eq_y]
    /-
      R : Type u
      inst✝² : CommSemiring R
      A : Type v
      inst✝¹ : CommSemiring A
      inst✝ : Algebra R A
      x : A
      I J : Submodule R A
      h : Membership.mem (HDiv.hDiv I J) x
      y : A
      x✝ : Membership.mem (HSMul.hSMul x ↑J) y
      y' : A
      hy' : Membership.mem (↑J) y'
      xy'_eq_y : Eq ((fun x_1 => HSMul.hSMul x x_1) y') y
      ⊢ Membership.mem (↑I) ((fun x_1 => HSMul.hSMul x x_1) y')
    -/
    apply h
    /-
      case a
      R : Type u
      inst✝² : CommSemiring R
      A : Type v
      inst✝¹ : CommSemiring A
      inst✝ : Algebra R A
      x : A
      I J : Submodule R A
      h : Membership.mem (HDiv.hDiv I J) x
      y : A
      x✝ : Membership.mem (HSMul.hSMul x ↑J) y
      y' : A
      hy' : Membership.mem (↑J) y'
      xy'_eq_y : Eq ((fun x_1 => HSMul.hSMul x x_1) y') y
      ⊢ Membership.mem J y'
    -/
    assumption, fun h _ hy => h (Set.smul_mem_smul_set hy)⟩
    /-
      🎉 no goals
    -/


theorem le_div_iff {I J K : Submodule R A} : I ≤ J / K ↔ ∀ x ∈ I, ∀ z ∈ K, x * z ∈ J :=
  Iff.refl _


theorem le_div_iff_mul_le {I J K : Submodule R A} : I ≤ J / K ↔ I * K ≤ J := by
  /-
    R : Type u
    inst✝² : CommSemiring R
    A : Type v
    inst✝¹ : CommSemiring A
    inst✝ : Algebra R A
    I J K : Submodule R A
    ⊢ Iff (LE.le I (HDiv.hDiv J K)) (LE.le (HMul.hMul I K) J)
  -/
  rw [le_div_iff, mul_le]
  /-
    🎉 no goals
  -/


theorem one_le_one_div {I : Submodule R A} : 1 ≤ 1 / I ↔ I ≤ 1 := by
  /-
    R : Type u
    inst✝² : CommSemiring R
    A : Type v
    inst✝¹ : CommSemiring A
    inst✝ : Algebra R A
    I : Submodule R A
    ⊢ Iff (LE.le 1 (HDiv.hDiv 1 I)) (LE.le I 1)
  -/
  constructor; all_goals intro hI
    /-
      case mp
      R : Type u
      inst✝² : CommSemiring R
      A : Type v
      inst✝¹ : CommSemiring A
      inst✝ : Algebra R A
      I : Submodule R A
      hI : LE.le 1 (HDiv.hDiv 1 I)
      ⊢ LE.le I 1
    -/
  · rwa [le_div_iff_mul_le, one_mul] at hI
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type u
      inst✝² : CommSemiring R
      A : Type v
      inst✝¹ : CommSemiring A
      inst✝ : Algebra R A
      I : Submodule R A
      hI : LE.le I 1
      ⊢ LE.le 1 (HDiv.hDiv 1 I)
    -/
  · rwa [le_div_iff_mul_le, one_mul]
    /-
      🎉 no goals
    -/


@[simp]
theorem one_mem_div {I J : Submodule R A} : 1 ∈ I / J ↔ J ≤ I := by
  /-
    R : Type u
    inst✝² : CommSemiring R
    A : Type v
    inst✝¹ : CommSemiring A
    inst✝ : Algebra R A
    I J : Submodule R A
    ⊢ Iff (Membership.mem (HDiv.hDiv I J) 1) (LE.le J I)
  -/
  rw [← one_le, le_div_iff_mul_le, one_mul]
  /-
    🎉 no goals
  -/


theorem le_self_mul_one_div {I : Submodule R A} (hI : I ≤ 1) : I ≤ I * (1 / I) := by
  /-
    R : Type u
    inst✝² : CommSemiring R
    A : Type v
    inst✝¹ : CommSemiring A
    inst✝ : Algebra R A
    I : Submodule R A
    hI : LE.le I 1
    ⊢ LE.le I (HMul.hMul I (HDiv.hDiv 1 I))
  -/
  refine (mul_one I).symm.trans_le ?_  -- Porting note: drop `rw {occs := _}` in favor of `refine`
  /-
    R : Type u
    inst✝² : CommSemiring R
    A : Type v
    inst✝¹ : CommSemiring A
    inst✝ : Algebra R A
    I : Submodule R A
    hI : LE.le I 1
    ⊢ LE.le (HMul.hMul I 1) (HMul.hMul I (HDiv.hDiv 1 I))
  -/
  apply mul_le_mul_right (one_le_one_div.mpr hI)
  /-
    🎉 no goals
  -/


theorem mul_one_div_le_one {I : Submodule R A} : I * (1 / I) ≤ 1 := by
  /-
    R : Type u
    inst✝² : CommSemiring R
    A : Type v
    inst✝¹ : CommSemiring A
    inst✝ : Algebra R A
    I : Submodule R A
    ⊢ LE.le (HMul.hMul I (HDiv.hDiv 1 I)) 1
  -/
  rw [Submodule.mul_le]
  /-
    R : Type u
    inst✝² : CommSemiring R
    A : Type v
    inst✝¹ : CommSemiring A
    inst✝ : Algebra R A
    I : Submodule R A
    ⊢ ∀ (m : A), Membership.mem I m → ∀ (n : A), Membership.mem (HDiv.hDiv 1 I) n  …
  -/
  intro m hm n hn
  /-
    R : Type u
    inst✝² : CommSemiring R
    A : Type v
    inst✝¹ : CommSemiring A
    inst✝ : Algebra R A
    I : Submodule R A
    m : A
    hm : Membership.mem I m
    n : A
    hn : Membership.mem (HDiv.hDiv 1 I) n
    ⊢ Membership.mem 1 (HMul.hMul m n)
  -/
  rw [Submodule.mem_div_iff_forall_mul_mem] at hn
  /-
    R : Type u
    inst✝² : CommSemiring R
    A : Type v
    inst✝¹ : CommSemiring A
    inst✝ : Algebra R A
    I : Submodule R A
    m : A
    hm : Membership.mem I m
    n : A
    hn : ∀ (y : A), Membership.mem I y → Membership.mem 1 (HMul.hMul n y)
    ⊢ Membership.mem 1 (HMul.hMul m n)
  -/
  rw [mul_comm]
  /-
    R : Type u
    inst✝² : CommSemiring R
    A : Type v
    inst✝¹ : CommSemiring A
    inst✝ : Algebra R A
    I : Submodule R A
    m : A
    hm : Membership.mem I m
    n : A
    hn : ∀ (y : A), Membership.mem I y → Membership.mem 1 (HMul.hMul n y)
    ⊢ Membership.mem 1 (HMul.hMul n m)
  -/
  exact hn m hm
  /-
    🎉 no goals
  -/


@[simp]
protected theorem map_div {B : Type*} [CommSemiring B] [Algebra R B] (I J : Submodule R A)
    (h : A ≃ₐ[R] B) : (I / J).map h.toLinearMap = I.map h.toLinearMap / J.map h.toLinearMap := by
  /-
    R : Type u
    inst✝⁴ : CommSemiring R
    A : Type v
    inst✝³ : CommSemiring A
    inst✝² : Algebra R A
    B : Type u_1
    inst✝¹ : CommSemiring B
    inst✝ : Algebra R B
    I J : Submodule R A
    h : AlgEquiv R A B
    ⊢ Eq (Submodule.map h.toLinearMap (HDiv.hDiv I J)) (HDiv.hDiv (Submodule.map h …
  -/
  ext x
  /-
    case h
    R : Type u
    inst✝⁴ : CommSemiring R
    A : Type v
    inst✝³ : CommSemiring A
    inst✝² : Algebra R A
    B : Type u_1
    inst✝¹ : CommSemiring B
    inst✝ : Algebra R B
    I J : Submodule R A
    h : AlgEquiv R A B
    x : B
    ⊢ Iff (Membership.mem (Submodule.map h.toLinearMap (HDiv.hDiv I J)) x) (Member …
  -/
  simp only [mem_map, mem_div_iff_forall_mul_mem]
  /-
    case h
    R : Type u
    inst✝⁴ : CommSemiring R
    A : Type v
    inst✝³ : CommSemiring A
    inst✝² : Algebra R A
    B : Type u_1
    inst✝¹ : CommSemiring B
    inst✝ : Algebra R B
    I J : Submodule R A
    h : AlgEquiv R A B
    x : B
    ⊢ Iff (Exists fun y => And (∀ (y_1 : A), Membership.mem J y_1 → Membership.mem …
  -/
  constructor
    /-
      case h.mp
      R : Type u
      inst✝⁴ : CommSemiring R
      A : Type v
      inst✝³ : CommSemiring A
      inst✝² : Algebra R A
      B : Type u_1
      inst✝¹ : CommSemiring B
      inst✝ : Algebra R B
      I J : Submodule R A
      h : AlgEquiv R A B
      x : B
      ⊢ (Exists fun y => And (∀ (y_1 : A), Membership.mem J y_1 → Membership.mem I ( …
    -/
  · rintro ⟨x, hx, rfl⟩ _ ⟨y, hy, rfl⟩
    /-
      case h.mp.intro.intro.intro.intro
      R : Type u
      inst✝⁴ : CommSemiring R
      A : Type v
      inst✝³ : CommSemiring A
      inst✝² : Algebra R A
      B : Type u_1
      inst✝¹ : CommSemiring B
      inst✝ : Algebra R B
      I J : Submodule R A
      h : AlgEquiv R A B
      x : A
      hx : ∀ (y : A), Membership.mem J y → Membership.mem I (HMul.hMul x y)
      y : A
      hy : Membership.mem J y
      ⊢ Exists fun y_1 => And (Membership.mem I y_1) (Eq (h.toLinearMap y_1) (HMul.h …
    -/
    exact ⟨x * y, hx _ hy, map_mul h x y⟩
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      R : Type u
      inst✝⁴ : CommSemiring R
      A : Type v
      inst✝³ : CommSemiring A
      inst✝² : Algebra R A
      B : Type u_1
      inst✝¹ : CommSemiring B
      inst✝ : Algebra R B
      I J : Submodule R A
      h : AlgEquiv R A B
      x : B
      ⊢ (∀ (y : B), (Exists fun y_1 => And (Membership.mem J y_1) (Eq (h.toLinearMap …
    -/
  · rintro hx
    /-
      case h.mpr
      R : Type u
      inst✝⁴ : CommSemiring R
      A : Type v
      inst✝³ : CommSemiring A
      inst✝² : Algebra R A
      B : Type u_1
      inst✝¹ : CommSemiring B
      inst✝ : Algebra R B
      I J : Submodule R A
      h : AlgEquiv R A B
      x : B
      hx : ∀ (y : B), (Exists fun y_1 => And (Membership.mem J y_1) (Eq (h.toLinearM …
      ⊢ Exists fun y => And (∀ (y_1 : A), Membership.mem J y_1 → Membership.mem I (H …
    -/
    refine ⟨h.symm x, fun z hz => ?_, h.apply_symm_apply x⟩
    /-
      case h.mpr
      R : Type u
      inst✝⁴ : CommSemiring R
      A : Type v
      inst✝³ : CommSemiring A
      inst✝² : Algebra R A
      B : Type u_1
      inst✝¹ : CommSemiring B
      inst✝ : Algebra R B
      I J : Submodule R A
      h : AlgEquiv R A B
      x : B
      hx : ∀ (y : B), (Exists fun y_1 => And (Membership.mem J y_1) (Eq (h.toLinearM …
      z : A
      hz : Membership.mem J z
      ⊢ Membership.mem I (HMul.hMul (h.symm x) z)
    -/
    obtain ⟨xz, xz_mem, hxz⟩ := hx (h z) ⟨z, hz, rfl⟩
    /-
      case h.mpr.intro.intro
      R : Type u
      inst✝⁴ : CommSemiring R
      A : Type v
      inst✝³ : CommSemiring A
      inst✝² : Algebra R A
      B : Type u_1
      inst✝¹ : CommSemiring B
      inst✝ : Algebra R B
      I J : Submodule R A
      h : AlgEquiv R A B
      x : B
      hx : ∀ (y : B), (Exists fun y_1 => And (Membership.mem J y_1) (Eq (h.toLinearM …
      z : A
      hz : Membership.mem J z
      xz : A
      xz_mem : Membership.mem I xz
      hxz : Eq (h.toLinearMap xz) (HMul.hMul x (h z))
      ⊢ Membership.mem I (HMul.hMul (h.symm x) z)
    -/
    convert xz_mem
    /-
      case h.e'_5
      R : Type u
      inst✝⁴ : CommSemiring R
      A : Type v
      inst✝³ : CommSemiring A
      inst✝² : Algebra R A
      B : Type u_1
      inst✝¹ : CommSemiring B
      inst✝ : Algebra R B
      I J : Submodule R A
      h : AlgEquiv R A B
      x : B
      hx : ∀ (y : B), (Exists fun y_1 => And (Membership.mem J y_1) (Eq (h.toLinearM …
      z : A
      hz : Membership.mem J z
      xz : A
      xz_mem : Membership.mem I xz
      hxz : Eq (h.toLinearMap xz) (HMul.hMul x (h z))
      ⊢ Eq (HMul.hMul (h.symm x) z) xz
    -/
    apply h.injective
    /-
      case h.e'_5.a
      R : Type u
      inst✝⁴ : CommSemiring R
      A : Type v
      inst✝³ : CommSemiring A
      inst✝² : Algebra R A
      B : Type u_1
      inst✝¹ : CommSemiring B
      inst✝ : Algebra R B
      I J : Submodule R A
      h : AlgEquiv R A B
      x : B
      hx : ∀ (y : B), (Exists fun y_1 => And (Membership.mem J y_1) (Eq (h.toLinearM …
      z : A
      hz : Membership.mem J z
      xz : A
      xz_mem : Membership.mem I xz
      hxz : Eq (h.toLinearMap xz) (HMul.hMul x (h z))
      ⊢ Eq (h (HMul.hMul (h.symm x) z)) (h xz)
    -/
    erw [map_mul, h.apply_symm_apply, hxz]
    /-
      🎉 no goals
    -/


