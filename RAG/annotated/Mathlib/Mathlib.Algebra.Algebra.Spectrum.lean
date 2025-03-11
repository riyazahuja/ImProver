local notation "↑ₐ" => algebraMap R A

-- definition and basic properties

/-- Given a commutative ring `R` and an `R`-algebra `A`, the *resolvent set* of `a : A`
is the `Set R` consisting of those `r : R` for which `r•1 - a` is a unit of the
algebra `A`. -/
def resolventSet (a : A) : Set R :=
  {r : R | IsUnit (↑ₐ r - a)}


/-- Given a commutative ring `R` and an `R`-algebra `A`, the *spectrum* of `a : A`
is the `Set R` consisting of those `r : R` for which `r•1 - a` is not a unit of the
algebra `A`.

The spectrum is simply the complement of the resolvent set. -/
def spectrum (a : A) : Set R :=
  (resolventSet R a)ᶜ


/-- Given an `a : A` where `A` is an `R`-algebra, the *resolvent* is
    a map `R → A` which sends `r : R` to `(algebraMap R A r - a)⁻¹` when
    `r ∈ resolvent R A` and `0` when `r ∈ spectrum R A`. -/
noncomputable def resolvent (a : A) (r : R) : A :=
  Ring.inverse (↑ₐ r - a)


/-- The unit `1 - r⁻¹ • a` constructed from `r • 1 - a` when the latter is a unit. -/
@[simps]
noncomputable def IsUnit.subInvSMul {r : Rˣ} {s : R} {a : A} (h : IsUnit <| r • ↑ₐ s - a) : Aˣ where
  val := ↑ₐ s - r⁻¹ • a
  inv := r • ↑h.unit⁻¹
                /-
                  R : Type u
                  A : Type v
                  inst✝² : CommSemiring R
                  inst✝¹ : Ring A
                  inst✝ : Algebra R A
                  r : Units R
                  s : R
                  a : A
                  h : IsUnit (HSub.hSub (HSMul.hSMul r ((algebraMap R A) s)) a)
                  ⊢ Eq (HMul.hMul (HSub.hSub ((algebraMap R A) s) (HSMul.hSMul (Inv.inv r) a)) ( …
                -/
  val_inv := by rw [mul_smul_comm, ← smul_mul_assoc, smul_sub, smul_inv_smul, h.mul_val_inv]
                /-
                  🎉 no goals
                -/
                /-
                  R : Type u
                  A : Type v
                  inst✝² : CommSemiring R
                  inst✝¹ : Ring A
                  inst✝ : Algebra R A
                  r : Units R
                  s : R
                  a : A
                  h : IsUnit (HSub.hSub (HSMul.hSMul r ((algebraMap R A) s)) a)
                  ⊢ Eq (HMul.hMul (HSMul.hSMul r ↑(Inv.inv h.unit)) (HSub.hSub ((algebraMap R A) …
                -/
  inv_val := by rw [smul_mul_assoc, ← mul_smul_comm, smul_sub, smul_inv_smul, h.val_inv_mul]
                /-
                  🎉 no goals
                -/


local notation "σ" => spectrum R


local notation "↑ₐ" => algebraMap R A


theorem mem_iff {r : R} {a : A} : r ∈ σ a ↔ ¬IsUnit (↑ₐ r - a) :=
  Iff.rfl


theorem not_mem_iff {r : R} {a : A} : r ∉ σ a ↔ IsUnit (↑ₐ r - a) := by
  /-
    R : Type u
    A : Type v
    inst✝² : CommSemiring R
    inst✝¹ : Ring A
    inst✝ : Algebra R A
    r : R
    a : A
    ⊢ Iff (Not (Membership.mem (spectrum R a) r)) (IsUnit (HSub.hSub ((algebraMap  …
  -/
  apply not_iff_not.mp
  /-
    R : Type u
    A : Type v
    inst✝² : CommSemiring R
    inst✝¹ : Ring A
    inst✝ : Algebra R A
    r : R
    a : A
    ⊢ Iff (Not (Not (Membership.mem (spectrum R a) r))) (Not (IsUnit (HSub.hSub (( …
  -/
  simp [Set.not_not_mem, mem_iff]
  /-
    🎉 no goals
  -/


theorem zero_mem_iff {a : A} : (0 : R) ∈ σ a ↔ ¬IsUnit a := by
  /-
    R : Type u
    A : Type v
    inst✝² : CommSemiring R
    inst✝¹ : Ring A
    inst✝ : Algebra R A
    a : A
    ⊢ Iff (Membership.mem (spectrum R a) 0) (Not (IsUnit a))
  -/
  rw [mem_iff, map_zero, zero_sub, IsUnit.neg_iff]
  /-
    🎉 no goals
  -/


alias ⟨not_isUnit_of_zero_mem, zero_mem⟩ := spectrum.zero_mem_iff


theorem zero_not_mem_iff {a : A} : (0 : R) ∉ σ a ↔ IsUnit a := by
  /-
    R : Type u
    A : Type v
    inst✝² : CommSemiring R
    inst✝¹ : Ring A
    inst✝ : Algebra R A
    a : A
    ⊢ Iff (Not (Membership.mem (spectrum R a) 0)) (IsUnit a)
  -/
  rw [zero_mem_iff, Classical.not_not]
  /-
    🎉 no goals
  -/


alias ⟨isUnit_of_zero_not_mem, zero_not_mem⟩ := spectrum.zero_not_mem_iff


@[simp]
lemma _root_.Units.zero_not_mem_spectrum (a : Aˣ) : 0 ∉ spectrum R (a : A) :=
  spectrum.zero_not_mem R a.isUnit


lemma subset_singleton_zero_compl {a : A} (ha : IsUnit a) : spectrum R a ⊆ {0}ᶜ :=
  Set.subset_compl_singleton_iff.mpr <| spectrum.zero_not_mem R ha


theorem mem_resolventSet_of_left_right_inverse {r : R} {a b c : A} (h₁ : (↑ₐ r - a) * b = 1)
    (h₂ : c * (↑ₐ r - a) = 1) : r ∈ resolventSet R a :=
                                    /-
                                      R : Type u
                                      A : Type v
                                      inst✝² : CommSemiring R
                                      inst✝¹ : Ring A
                                      inst✝ : Algebra R A
                                      r : R
                                      a b c : A
                                      h₁ : Eq (HMul.hMul (HSub.hSub ((algebraMap R A) r) a) b) 1
                                      h₂ : Eq (HMul.hMul c (HSub.hSub ((algebraMap R A) r) a)) 1
                                      ⊢ Eq (HMul.hMul b (HSub.hSub ((algebraMap R A) r) a)) 1
                                    -/
  Units.isUnit ⟨↑ₐ r - a, b, h₁, by rwa [← left_inv_eq_right_inv h₂ h₁]⟩
                                    /-
                                      🎉 no goals
                                    -/


theorem mem_resolventSet_iff {r : R} {a : A} : r ∈ resolventSet R a ↔ IsUnit (↑ₐ r - a) :=
  Iff.rfl


@[simp]
theorem algebraMap_mem_iff (S : Type*) {R A : Type*} [CommSemiring R] [CommSemiring S]
    [Ring A] [Algebra R S] [Algebra R A] [Algebra S A] [IsScalarTower R S A] {a : A} {r : R} :
    algebraMap R S r ∈ spectrum S a ↔ r ∈ spectrum R a := by
  /-
    S : Type u_1
    R : Type u_2
    A : Type u_3
    inst✝⁶ : CommSemiring R
    inst✝⁵ : CommSemiring S
    inst✝⁴ : Ring A
    inst✝³ : Algebra R S
    inst✝² : Algebra R A
    inst✝¹ : Algebra S A
    inst✝ : IsScalarTower R S A
    a : A
    r : R
    ⊢ Iff (Membership.mem (spectrum S a) ((algebraMap R S) r)) (Membership.mem (sp …
  -/
  simp only [spectrum.mem_iff, Algebra.algebraMap_eq_smul_one, smul_assoc, one_smul]
  /-
    🎉 no goals
  -/


protected alias ⟨of_algebraMap_mem, algebraMap_mem⟩ := spectrum.algebraMap_mem_iff


@[simp]
theorem preimage_algebraMap (S : Type*) {R A : Type*} [CommSemiring R] [CommSemiring S]
    [Ring A] [Algebra R S] [Algebra R A] [Algebra S A] [IsScalarTower R S A] {a : A} :
    algebraMap R S ⁻¹' spectrum S a = spectrum R a :=
  Set.ext fun _ => spectrum.algebraMap_mem_iff _


@[simp]
theorem resolventSet_of_subsingleton [Subsingleton A] (a : A) : resolventSet R a = Set.univ := by
  /-
    R : Type u
    A : Type v
    inst✝³ : CommSemiring R
    inst✝² : Ring A
    inst✝¹ : Algebra R A
    inst✝ : Subsingleton A
    a : A
    ⊢ Eq (resolventSet R a) Set.univ
  -/
  simp_rw [resolventSet, Subsingleton.elim (algebraMap R A _ - a) 1, isUnit_one, Set.setOf_true]
  /-
    🎉 no goals
  -/


@[simp]
theorem of_subsingleton [Subsingleton A] (a : A) : spectrum R a = ∅ := by
  /-
    R : Type u
    A : Type v
    inst✝³ : CommSemiring R
    inst✝² : Ring A
    inst✝¹ : Algebra R A
    inst✝ : Subsingleton A
    a : A
    ⊢ Eq (spectrum R a) EmptyCollection.emptyCollection
  -/
  rw [spectrum, resolventSet_of_subsingleton, Set.compl_univ]
  /-
    🎉 no goals
  -/


theorem resolvent_eq {a : A} {r : R} (h : r ∈ resolventSet R a) : resolvent a r = ↑h.unit⁻¹ :=
  Ring.inverse_unit h.unit


theorem units_smul_resolvent {r : Rˣ} {s : R} {a : A} :
    r • resolvent a (s : R) = resolvent (r⁻¹ • a) (r⁻¹ • s : R) := by
  /-
    R : Type u
    A : Type v
    inst✝² : CommSemiring R
    inst✝¹ : Ring A
    inst✝ : Algebra R A
    r : Units R
    s : R
    a : A
    ⊢ Eq (HSMul.hSMul r (resolvent a s)) (resolvent (HSMul.hSMul (Inv.inv r) a) (H …
  -/
  by_cases h : s ∈ spectrum R a
    /-
      case pos
      R : Type u
      A : Type v
      inst✝² : CommSemiring R
      inst✝¹ : Ring A
      inst✝ : Algebra R A
      r : Units R
      s : R
      a : A
      h : Membership.mem (spectrum R a) s
      ⊢ Eq (HSMul.hSMul r (resolvent a s)) (resolvent (HSMul.hSMul (Inv.inv r) a) (H …
    -/
  · rw [mem_iff] at h
    /-
      case pos
      R : Type u
      A : Type v
      inst✝² : CommSemiring R
      inst✝¹ : Ring A
      inst✝ : Algebra R A
      r : Units R
      s : R
      a : A
      h : Not (IsUnit (HSub.hSub ((algebraMap R A) s) a))
      ⊢ Eq (HSMul.hSMul r (resolvent a s)) (resolvent (HSMul.hSMul (Inv.inv r) a) (H …
    -/
    simp only [resolvent, Algebra.algebraMap_eq_smul_one] at *
    /-
      case pos
      R : Type u
      A : Type v
      inst✝² : CommSemiring R
      inst✝¹ : Ring A
      inst✝ : Algebra R A
      r : Units R
      s : R
      a : A
      h : Not (IsUnit (HSub.hSub (HSMul.hSMul s 1) a))
      ⊢ Eq (HSMul.hSMul r (Ring.inverse (HSub.hSub (HSMul.hSMul s 1) a))) (Ring.inve …
    -/
    rw [smul_assoc, ← smul_sub]
    have h' : ¬IsUnit (r⁻¹ • (s • (1 : A) - a)) := fun hu =>
      h (by simpa only [smul_inv_smul] using IsUnit.smul r hu)
    /-
      case pos
      R : Type u
      A : Type v
      inst✝² : CommSemiring R
      inst✝¹ : Ring A
      inst✝ : Algebra R A
      r : Units R
      s : R
      a : A
      h : Not (IsUnit (HSub.hSub (HSMul.hSMul s 1) a))
      h' : Not (IsUnit (HSMul.hSMul (Inv.inv r) (HSub.hSub (HSMul.hSMul s 1) a)))
      ⊢ Eq (HSMul.hSMul r (Ring.inverse (HSub.hSub (HSMul.hSMul s 1) a))) (Ring.inve …
    -/
    simp only [Ring.inverse_non_unit _ h, Ring.inverse_non_unit _ h', smul_zero]
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u
      A : Type v
      inst✝² : CommSemiring R
      inst✝¹ : Ring A
      inst✝ : Algebra R A
      r : Units R
      s : R
      a : A
      h : Not (Membership.mem (spectrum R a) s)
      ⊢ Eq (HSMul.hSMul r (resolvent a s)) (resolvent (HSMul.hSMul (Inv.inv r) a) (H …
    -/
  · simp only [resolvent]
    have h' : IsUnit (r • algebraMap R A (r⁻¹ • s) - a) := by
      simpa [Algebra.algebraMap_eq_smul_one, smul_assoc] using not_mem_iff.mp h
    rw [← h'.val_subInvSMul, ← (not_mem_iff.mp h).unit_spec, Ring.inverse_unit, Ring.inverse_unit,
      h'.val_inv_subInvSMul]
    /-
      case neg
      R : Type u
      A : Type v
      inst✝² : CommSemiring R
      inst✝¹ : Ring A
      inst✝ : Algebra R A
      r : Units R
      s : R
      a : A
      h : Not (Membership.mem (spectrum R a) s)
      h' : IsUnit (HSub.hSub (HSMul.hSMul r ((algebraMap R A) (HSMul.hSMul (Inv.inv  …
      ⊢ Eq (HSMul.hSMul r ↑(Inv.inv ⋯.unit)) (HSMul.hSMul r ↑(Inv.inv h'.unit))
    -/
    simp only [Algebra.algebraMap_eq_smul_one, smul_assoc, smul_inv_smul]
    /-
      🎉 no goals
    -/


theorem units_smul_resolvent_self {r : Rˣ} {a : A} :
    r • resolvent a (r : R) = resolvent (r⁻¹ • a) (1 : R) := by
  simpa only [Units.smul_def, Algebra.id.smul_eq_mul, Units.inv_mul] using
    @units_smul_resolvent _ _ _ _ _ r r a


/-- The resolvent is a unit when the argument is in the resolvent set. -/
theorem isUnit_resolvent {r : R} {a : A} : r ∈ resolventSet R a ↔ IsUnit (resolvent a r) :=
  isUnit_ring_inverse.symm


theorem inv_mem_resolventSet {r : Rˣ} {a : Aˣ} (h : (r : R) ∈ resolventSet R (a : A)) :
    (↑r⁻¹ : R) ∈ resolventSet R (↑a⁻¹ : A) := by
  /-
    R : Type u
    A : Type v
    inst✝² : CommSemiring R
    inst✝¹ : Ring A
    inst✝ : Algebra R A
    r : Units R
    a : Units A
    h : Membership.mem (resolventSet R ↑a) ↑r
    ⊢ Membership.mem (resolventSet R ↑(Inv.inv a)) ↑(Inv.inv r)
  -/
  rw [mem_resolventSet_iff, Algebra.algebraMap_eq_smul_one, ← Units.smul_def] at h ⊢
  /-
    R : Type u
    A : Type v
    inst✝² : CommSemiring R
    inst✝¹ : Ring A
    inst✝ : Algebra R A
    r : Units R
    a : Units A
    h : IsUnit (HSub.hSub (HSMul.hSMul r 1) ↑a)
    ⊢ IsUnit (HSub.hSub (HSMul.hSMul (Inv.inv r) 1) ↑(Inv.inv a))
  -/
  rw [IsUnit.smul_sub_iff_sub_inv_smul, inv_inv, IsUnit.sub_iff]
  have h₁ : (a : A) * (r • (↑a⁻¹ : A) - 1) = r • (1 : A) - a := by
    rw [mul_sub, mul_smul_comm, a.mul_inv, mul_one]
  have h₂ : (r • (↑a⁻¹ : A) - 1) * a = r • (1 : A) - a := by
    rw [sub_mul, smul_mul_assoc, a.inv_mul, one_mul]
  /-
    R : Type u
    A : Type v
    inst✝² : CommSemiring R
    inst✝¹ : Ring A
    inst✝ : Algebra R A
    r : Units R
    a : Units A
    h : IsUnit (HSub.hSub (HSMul.hSMul r 1) ↑a)
    h₁ : Eq (HMul.hMul (↑a) (HSub.hSub (HSMul.hSMul r ↑(Inv.inv a)) 1)) (HSub.hSub …
    h₂ : Eq (HMul.hMul (HSub.hSub (HSMul.hSMul r ↑(Inv.inv a)) 1) ↑a) (HSub.hSub ( …
    ⊢ IsUnit (HSub.hSub (HSMul.hSMul r ↑(Inv.inv a)) 1)
  -/
  have hcomm : Commute (a : A) (r • (↑a⁻¹ : A) - 1) := by rwa [← h₂] at h₁
  /-
    R : Type u
    A : Type v
    inst✝² : CommSemiring R
    inst✝¹ : Ring A
    inst✝ : Algebra R A
    r : Units R
    a : Units A
    h : IsUnit (HSub.hSub (HSMul.hSMul r 1) ↑a)
    h₁ : Eq (HMul.hMul (↑a) (HSub.hSub (HSMul.hSMul r ↑(Inv.inv a)) 1)) (HSub.hSub …
    h₂ : Eq (HMul.hMul (HSub.hSub (HSMul.hSMul r ↑(Inv.inv a)) 1) ↑a) (HSub.hSub ( …
    hcomm : Commute (↑a) (HSub.hSub (HSMul.hSMul r ↑(Inv.inv a)) 1)
    ⊢ IsUnit (HSub.hSub (HSMul.hSMul r ↑(Inv.inv a)) 1)
  -/
  exact (hcomm.isUnit_mul_iff.mp (h₁.symm ▸ h)).2
  /-
    🎉 no goals
  -/


theorem inv_mem_iff {r : Rˣ} {a : Aˣ} : (r : R) ∈ σ (a : A) ↔ (↑r⁻¹ : R) ∈ σ (↑a⁻¹ : A) :=
  not_iff_not.2 <| ⟨inv_mem_resolventSet, inv_mem_resolventSet⟩


theorem zero_mem_resolventSet_of_unit (a : Aˣ) : 0 ∈ resolventSet R (a : A) := by
  /-
    R : Type u
    A : Type v
    inst✝² : CommSemiring R
    inst✝¹ : Ring A
    inst✝ : Algebra R A
    a : Units A
    ⊢ Membership.mem (resolventSet R ↑a) 0
  -/
  simpa only [mem_resolventSet_iff, ← not_mem_iff, zero_not_mem_iff] using a.isUnit
  /-
    🎉 no goals
  -/


theorem ne_zero_of_mem_of_unit {a : Aˣ} {r : R} (hr : r ∈ σ (a : A)) : r ≠ 0 := fun hn =>
  (hn ▸ hr) (zero_mem_resolventSet_of_unit a)


theorem add_mem_iff {a : A} {r s : R} : r + s ∈ σ a ↔ r ∈ σ (-↑ₐ s + a) := by
  /-
    R : Type u
    A : Type v
    inst✝² : CommSemiring R
    inst✝¹ : Ring A
    inst✝ : Algebra R A
    a : A
    r s : R
    ⊢ Iff (Membership.mem (spectrum R a) (HAdd.hAdd r s)) (Membership.mem (spectru …
  -/
  simp only [mem_iff, sub_neg_eq_add, ← sub_sub, map_add]
  /-
    🎉 no goals
  -/


theorem add_mem_add_iff {a : A} {r s : R} : r + s ∈ σ (↑ₐ s + a) ↔ r ∈ σ a := by
  /-
    R : Type u
    A : Type v
    inst✝² : CommSemiring R
    inst✝¹ : Ring A
    inst✝ : Algebra R A
    a : A
    r s : R
    ⊢ Iff (Membership.mem (spectrum R (HAdd.hAdd ((algebraMap R A) s) a)) (HAdd.hA …
  -/
  rw [add_mem_iff, neg_add_cancel_left]
  /-
    🎉 no goals
  -/


theorem smul_mem_smul_iff {a : A} {s : R} {r : Rˣ} : r • s ∈ σ (r • a) ↔ s ∈ σ a := by
  simp only [mem_iff, not_iff_not, Algebra.algebraMap_eq_smul_one, smul_assoc, ← smul_sub,
    isUnit_smul_iff]


theorem unit_smul_eq_smul (a : A) (r : Rˣ) : σ (r • a) = r • σ a := by
  /-
    R : Type u
    A : Type v
    inst✝² : CommSemiring R
    inst✝¹ : Ring A
    inst✝ : Algebra R A
    a : A
    r : Units R
    ⊢ Eq (spectrum R (HSMul.hSMul r a)) (HSMul.hSMul r (spectrum R a))
  -/
  ext x
  /-
    case h
    R : Type u
    A : Type v
    inst✝² : CommSemiring R
    inst✝¹ : Ring A
    inst✝ : Algebra R A
    a : A
    r : Units R
    x : R
    ⊢ Iff (Membership.mem (spectrum R (HSMul.hSMul r a)) x) (Membership.mem (HSMul …
  -/
  have x_eq : x = r • r⁻¹ • x := by simp
  /-
    case h
    R : Type u
    A : Type v
    inst✝² : CommSemiring R
    inst✝¹ : Ring A
    inst✝ : Algebra R A
    a : A
    r : Units R
    x : R
    x_eq : Eq x (HSMul.hSMul r (HSMul.hSMul (Inv.inv r) x))
    ⊢ Iff (Membership.mem (spectrum R (HSMul.hSMul r a)) x) (Membership.mem (HSMul …
  -/
  nth_rw 1 [x_eq]
  /-
    case h
    R : Type u
    A : Type v
    inst✝² : CommSemiring R
    inst✝¹ : Ring A
    inst✝ : Algebra R A
    a : A
    r : Units R
    x : R
    x_eq : Eq x (HSMul.hSMul r (HSMul.hSMul (Inv.inv r) x))
    ⊢ Iff (Membership.mem (spectrum R (HSMul.hSMul r a)) (HSMul.hSMul r (HSMul.hSM …
  -/
  rw [smul_mem_smul_iff]
  /-
    case h
    R : Type u
    A : Type v
    inst✝² : CommSemiring R
    inst✝¹ : Ring A
    inst✝ : Algebra R A
    a : A
    r : Units R
    x : R
    x_eq : Eq x (HSMul.hSMul r (HSMul.hSMul (Inv.inv r) x))
    ⊢ Iff (Membership.mem (spectrum R a) (HSMul.hSMul (Inv.inv r) x)) (Membership. …
  -/
  constructor
    /-
      case h.mp
      R : Type u
      A : Type v
      inst✝² : CommSemiring R
      inst✝¹ : Ring A
      inst✝ : Algebra R A
      a : A
      r : Units R
      x : R
      x_eq : Eq x (HSMul.hSMul r (HSMul.hSMul (Inv.inv r) x))
      ⊢ Membership.mem (spectrum R a) (HSMul.hSMul (Inv.inv r) x) → Membership.mem ( …
    -/
  · exact fun h => ⟨r⁻¹ • x, ⟨h, show r • r⁻¹ • x = x by simp⟩⟩
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      R : Type u
      A : Type v
      inst✝² : CommSemiring R
      inst✝¹ : Ring A
      inst✝ : Algebra R A
      a : A
      r : Units R
      x : R
      x_eq : Eq x (HSMul.hSMul r (HSMul.hSMul (Inv.inv r) x))
      ⊢ Membership.mem (HSMul.hSMul r (spectrum R a)) x → Membership.mem (spectrum R …
    -/
  · rintro ⟨w, _, (x'_eq : r • w = x)⟩
    /-
      case h.mpr.intro.intro
      R : Type u
      A : Type v
      inst✝² : CommSemiring R
      inst✝¹ : Ring A
      inst✝ : Algebra R A
      a : A
      r : Units R
      x : R
      x_eq : Eq x (HSMul.hSMul r (HSMul.hSMul (Inv.inv r) x))
      w : R
      left✝ : Membership.mem (spectrum R a) w
      x'_eq : Eq (HSMul.hSMul r w) x
      ⊢ Membership.mem (spectrum R a) (HSMul.hSMul (Inv.inv r) x)
    -/
    simpa [← x'_eq ]
    /-
      🎉 no goals
    -/

-- `r ∈ σ(a*b) ↔ r ∈ σ(b*a)` for any `r : Rˣ`

theorem unit_mem_mul_iff_mem_swap_mul {a b : A} {r : Rˣ} : ↑r ∈ σ (a * b) ↔ ↑r ∈ σ (b * a) := by
  have h₁ : ∀ x y : A, IsUnit (1 - x * y) → IsUnit (1 - y * x) := by
    refine fun x y h => ⟨⟨1 - y * x, 1 + y * h.unit.inv * x, ?_, ?_⟩, rfl⟩
    · calc
        (1 - y * x) * (1 + y * (IsUnit.unit h).inv * x) =
            1 - y * x + y * ((1 - x * y) * h.unit.inv) * x := by noncomm_ring
        _ = 1 := by simp only [Units.inv_eq_val_inv, IsUnit.mul_val_inv, mul_one, sub_add_cancel]
    · calc
        (1 + y * (IsUnit.unit h).inv * x) * (1 - y * x) =
            1 - y * x + y * (h.unit.inv * (1 - x * y)) * x := by noncomm_ring
        _ = 1 := by simp only [Units.inv_eq_val_inv, IsUnit.val_inv_mul, mul_one, sub_add_cancel]
  /-
    R : Type u
    A : Type v
    inst✝² : CommSemiring R
    inst✝¹ : Ring A
    inst✝ : Algebra R A
    a b : A
    r : Units R
    h₁ : ∀ (x y : A), IsUnit (HSub.hSub 1 (HMul.hMul x y)) → IsUnit (HSub.hSub 1 ( …
    ⊢ Iff (Membership.mem (spectrum R (HMul.hMul a b)) ↑r) (Membership.mem (spectr …
  -/
  have := Iff.intro (h₁ (r⁻¹ • a) b) (h₁ b (r⁻¹ • a))
  /-
    R : Type u
    A : Type v
    inst✝² : CommSemiring R
    inst✝¹ : Ring A
    inst✝ : Algebra R A
    a b : A
    r : Units R
    h₁ : ∀ (x y : A), IsUnit (HSub.hSub 1 (HMul.hMul x y)) → IsUnit (HSub.hSub 1 ( …
    this : Iff (IsUnit (HSub.hSub 1 (HMul.hMul (HSMul.hSMul (Inv.inv r) a) b))) (I …
    ⊢ Iff (Membership.mem (spectrum R (HMul.hMul a b)) ↑r) (Membership.mem (spectr …
  -/
  rw [mul_smul_comm r⁻¹ b a] at this
  simpa only [mem_iff, not_iff_not, Algebra.algebraMap_eq_smul_one, ← Units.smul_def,
    IsUnit.smul_sub_iff_sub_inv_smul, smul_mul_assoc]


theorem preimage_units_mul_eq_swap_mul {a b : A} :
    ((↑) : Rˣ → R) ⁻¹' σ (a * b) = (↑) ⁻¹' σ (b * a) :=
  Set.ext fun _ => unit_mem_mul_iff_mem_swap_mul


theorem star_mem_resolventSet_iff {r : R} {a : A} :
    star r ∈ resolventSet R a ↔ r ∈ resolventSet R (star a) := by
  /-
    R : Type u
    A : Type v
    inst✝⁵ : CommSemiring R
    inst✝⁴ : Ring A
    inst✝³ : Algebra R A
    inst✝² : InvolutiveStar R
    inst✝¹ : StarRing A
    inst✝ : StarModule R A
    r : R
    a : A
    ⊢ Iff (Membership.mem (resolventSet R a) (Star.star r)) (Membership.mem (resol …
  -/
  refine ⟨fun h => ?_, fun h => ?_⟩ <;>
    simpa only [mem_resolventSet_iff, Algebra.algebraMap_eq_smul_one, star_sub, star_smul,
      star_star, star_one] using IsUnit.star h


protected theorem map_star (a : A) : σ (star a) = star (σ a) := by
  /-
    R : Type u
    A : Type v
    inst✝⁵ : CommSemiring R
    inst✝⁴ : Ring A
    inst✝³ : Algebra R A
    inst✝² : InvolutiveStar R
    inst✝¹ : StarRing A
    inst✝ : StarModule R A
    a : A
    ⊢ Eq (spectrum R (Star.star a)) (Star.star (spectrum R a))
  -/
  ext
  /-
    case h
    R : Type u
    A : Type v
    inst✝⁵ : CommSemiring R
    inst✝⁴ : Ring A
    inst✝³ : Algebra R A
    inst✝² : InvolutiveStar R
    inst✝¹ : StarRing A
    inst✝ : StarModule R A
    a : A
    x✝ : R
    ⊢ Iff (Membership.mem (spectrum R (Star.star a)) x✝) (Membership.mem (Star.sta …
  -/
  simpa only [Set.mem_star, mem_iff, not_iff_not] using star_mem_resolventSet_iff.symm
  /-
    🎉 no goals
  -/


theorem subset_subalgebra {S R A : Type*} [CommSemiring R] [Ring A] [Algebra R A]
    [SetLike S A] [SubringClass S A] [SMulMemClass S R A] {s : S} (a : s) :
    spectrum R (a : A) ⊆ spectrum R a :=
  Set.compl_subset_compl.mpr fun _ ↦ IsUnit.map (SubalgebraClass.val s)


@[deprecated subset_subalgebra (since := "2024-07-19")]
theorem subset_starSubalgebra [StarRing R] [StarRing A] [StarModule R A] {S : StarSubalgebra R A}
    (a : S) : spectrum R (a : A) ⊆ spectrum R a :=
  subset_subalgebra a


theorem singleton_add_eq (a : A) (r : R) : {r} + σ a = σ (↑ₐ r + a) :=
  ext fun x => by
    /-
      R : Type u
      A : Type v
      inst✝² : CommRing R
      inst✝¹ : Ring A
      inst✝ : Algebra R A
      a : A
      r x : R
      ⊢ Iff (Membership.mem (HAdd.hAdd (Singleton.singleton r) (spectrum R a)) x) (M …
    -/
    rw [singleton_add, image_add_left, mem_preimage, add_comm, add_mem_iff, map_neg, neg_neg]
    /-
      🎉 no goals
    -/


theorem add_singleton_eq (a : A) (r : R) : σ a + {r} = σ (a + ↑ₐ r) :=
  add_comm {r} (σ a) ▸ add_comm (algebraMap R A r) a ▸ singleton_add_eq a r


theorem vadd_eq (a : A) (r : R) : r +ᵥ σ a = σ (↑ₐ r + a) :=
  singleton_add.symm.trans <| singleton_add_eq a r


theorem neg_eq (a : A) : -σ a = σ (-a) :=
  Set.ext fun x => by
    /-
      R : Type u
      A : Type v
      inst✝² : CommRing R
      inst✝¹ : Ring A
      inst✝ : Algebra R A
      a : A
      x : R
      ⊢ Iff (Membership.mem (Neg.neg (spectrum R a)) x) (Membership.mem (spectrum R  …
    -/
    simp only [mem_neg, mem_iff, map_neg, ← neg_add', IsUnit.neg_iff, sub_neg_eq_add]
    /-
      🎉 no goals
    -/


theorem singleton_sub_eq (a : A) (r : R) : {r} - σ a = σ (↑ₐ r - a) := by
  /-
    R : Type u
    A : Type v
    inst✝² : CommRing R
    inst✝¹ : Ring A
    inst✝ : Algebra R A
    a : A
    r : R
    ⊢ Eq (HSub.hSub (Singleton.singleton r) (spectrum R a)) (spectrum R (HSub.hSub …
  -/
  rw [sub_eq_add_neg, neg_eq, singleton_add_eq, sub_eq_add_neg]
  /-
    🎉 no goals
  -/


theorem sub_singleton_eq (a : A) (r : R) : σ a - {r} = σ (a - ↑ₐ r) := by
  /-
    R : Type u
    A : Type v
    inst✝² : CommRing R
    inst✝¹ : Ring A
    inst✝ : Algebra R A
    a : A
    r : R
    ⊢ Eq (HSub.hSub (spectrum R a) (Singleton.singleton r)) (spectrum R (HSub.hSub …
  -/
  simpa only [neg_sub, neg_eq] using congr_arg Neg.neg (singleton_sub_eq a r)
  /-
    🎉 no goals
  -/


@[simp]
lemma inv₀_mem_iff {r : R} {a : Aˣ} :
    r⁻¹ ∈ spectrum R (a : A) ↔ r ∈ spectrum R (↑a⁻¹ : A) := by
  /-
    R : Type u
    A : Type v
    inst✝² : Semifield R
    inst✝¹ : Ring A
    inst✝ : Algebra R A
    r : R
    a : Units A
    ⊢ Iff (Membership.mem (spectrum R ↑a) (Inv.inv r)) (Membership.mem (spectrum R …
  -/
  obtain (rfl | hr) := eq_or_ne r 0
    /-
      case inl
      R : Type u
      A : Type v
      inst✝² : Semifield R
      inst✝¹ : Ring A
      inst✝ : Algebra R A
      a : Units A
      ⊢ Iff (Membership.mem (spectrum R ↑a) (Inv.inv 0)) (Membership.mem (spectrum R …
    -/
  · simp [zero_mem_iff]
    /-
      🎉 no goals
    -/
    /-
      case inr
      R : Type u
      A : Type v
      inst✝² : Semifield R
      inst✝¹ : Ring A
      inst✝ : Algebra R A
      r : R
      a : Units A
      hr : Ne r 0
      ⊢ Iff (Membership.mem (spectrum R ↑a) (Inv.inv r)) (Membership.mem (spectrum R …
    -/
  · lift r to Rˣ using hr.isUnit
    /-
      case inr.intro
      R : Type u
      A : Type v
      inst✝² : Semifield R
      inst✝¹ : Ring A
      inst✝ : Algebra R A
      a : Units A
      r : Units R
      hr : Ne (↑r) 0
      ⊢ Iff (Membership.mem (spectrum R ↑a) (Inv.inv ↑r)) (Membership.mem (spectrum  …
    -/
    simp [inv_mem_iff]
    /-
      🎉 no goals
    -/


lemma inv₀_mem_inv_iff {r : R} {a : Aˣ} :
    r⁻¹ ∈ spectrum R (↑a⁻¹ : A) ↔ r ∈ spectrum R (a : A) := by
  /-
    R : Type u
    A : Type v
    inst✝² : Semifield R
    inst✝¹ : Ring A
    inst✝ : Algebra R A
    r : R
    a : Units A
    ⊢ Iff (Membership.mem (spectrum R ↑(Inv.inv a)) (Inv.inv r)) (Membership.mem ( …
  -/
  simp
  /-
    🎉 no goals
  -/


alias ⟨of_inv₀_mem, inv₀_mem⟩ := inv₀_mem_iff

alias ⟨of_inv₀_mem_inv, inv₀_mem_inv⟩ := inv₀_mem_inv_iff


local notation "σ" => spectrum 𝕜


local notation "↑ₐ" => algebraMap 𝕜 A


/-- Without the assumption `Nontrivial A`, then `0 : A` would be invertible. -/
@[simp]
theorem zero_eq [Nontrivial A] : σ (0 : A) = {0} := by
  /-
    𝕜 : Type u
    A : Type v
    inst✝³ : Field 𝕜
    inst✝² : Ring A
    inst✝¹ : Algebra 𝕜 A
    inst✝ : Nontrivial A
    ⊢ Eq (spectrum 𝕜 0) (Singleton.singleton 0)
  -/
  refine Set.Subset.antisymm ?_ (by simp [Algebra.algebraMap_eq_smul_one, mem_iff])
  /-
    𝕜 : Type u
    A : Type v
    inst✝³ : Field 𝕜
    inst✝² : Ring A
    inst✝¹ : Algebra 𝕜 A
    inst✝ : Nontrivial A
    ⊢ HasSubset.Subset (spectrum 𝕜 0) (Singleton.singleton 0)
  -/
  rw [spectrum, Set.compl_subset_comm]
  /-
    𝕜 : Type u
    A : Type v
    inst✝³ : Field 𝕜
    inst✝² : Ring A
    inst✝¹ : Algebra 𝕜 A
    inst✝ : Nontrivial A
    ⊢ HasSubset.Subset (HasCompl.compl (Singleton.singleton 0)) (resolventSet 𝕜 0)
  -/
  intro k hk
  /-
    𝕜 : Type u
    A : Type v
    inst✝³ : Field 𝕜
    inst✝² : Ring A
    inst✝¹ : Algebra 𝕜 A
    inst✝ : Nontrivial A
    k : 𝕜
    hk : Membership.mem (HasCompl.compl (Singleton.singleton 0)) k
    ⊢ Membership.mem (resolventSet 𝕜 0) k
  -/
  rw [Set.mem_compl_singleton_iff] at hk
  /-
    𝕜 : Type u
    A : Type v
    inst✝³ : Field 𝕜
    inst✝² : Ring A
    inst✝¹ : Algebra 𝕜 A
    inst✝ : Nontrivial A
    k : 𝕜
    hk : Ne k 0
    ⊢ Membership.mem (resolventSet 𝕜 0) k
  -/
  have : IsUnit (Units.mk0 k hk • (1 : A)) := IsUnit.smul (Units.mk0 k hk) isUnit_one
  /-
    𝕜 : Type u
    A : Type v
    inst✝³ : Field 𝕜
    inst✝² : Ring A
    inst✝¹ : Algebra 𝕜 A
    inst✝ : Nontrivial A
    k : 𝕜
    hk : Ne k 0
    this : IsUnit (HSMul.hSMul (Units.mk0 k hk) 1)
    ⊢ Membership.mem (resolventSet 𝕜 0) k
  -/
  simpa [mem_resolventSet_iff, Algebra.algebraMap_eq_smul_one]
  /-
    🎉 no goals
  -/


@[simp]
theorem scalar_eq [Nontrivial A] (k : 𝕜) : σ (↑ₐ k) = {k} := by
  /-
    𝕜 : Type u
    A : Type v
    inst✝³ : Field 𝕜
    inst✝² : Ring A
    inst✝¹ : Algebra 𝕜 A
    inst✝ : Nontrivial A
    k : 𝕜
    ⊢ Eq (spectrum 𝕜 ((algebraMap 𝕜 A) k)) (Singleton.singleton k)
  -/
  rw [← add_zero (↑ₐ k), ← singleton_add_eq, zero_eq, Set.singleton_add_singleton, add_zero]
  /-
    🎉 no goals
  -/


@[simp]
theorem one_eq [Nontrivial A] : σ (1 : A) = {1} :=
  calc
                               /-
                                 𝕜 : Type u
                                 A : Type v
                                 inst✝³ : Field 𝕜
                                 inst✝² : Ring A
                                 inst✝¹ : Algebra 𝕜 A
                                 inst✝ : Nontrivial A
                                 ⊢ Eq (spectrum 𝕜 1) (spectrum 𝕜 ((algebraMap 𝕜 A) 1))
                               -/
    σ (1 : A) = σ (↑ₐ 1) := by rw [Algebra.algebraMap_eq_smul_one, one_smul]
                               /-
                                 🎉 no goals
                               -/
    _ = {1} := scalar_eq 1


/-- the assumption `(σ a).Nonempty` is necessary and cannot be removed without
further conditions on the algebra `A` and scalar field `𝕜`. -/
theorem smul_eq_smul [Nontrivial A] (k : 𝕜) (a : A) (ha : (σ a).Nonempty) :
    σ (k • a) = k • σ a := by
  /-
    𝕜 : Type u
    A : Type v
    inst✝³ : Field 𝕜
    inst✝² : Ring A
    inst✝¹ : Algebra 𝕜 A
    inst✝ : Nontrivial A
    k : 𝕜
    a : A
    ha : (spectrum 𝕜 a).Nonempty
    ⊢ Eq (spectrum 𝕜 (HSMul.hSMul k a)) (HSMul.hSMul k (spectrum 𝕜 a))
  -/
  rcases eq_or_ne k 0 with (rfl | h)
    /-
      case inl
      𝕜 : Type u
      A : Type v
      inst✝³ : Field 𝕜
      inst✝² : Ring A
      inst✝¹ : Algebra 𝕜 A
      inst✝ : Nontrivial A
      a : A
      ha : (spectrum 𝕜 a).Nonempty
      ⊢ Eq (spectrum 𝕜 (HSMul.hSMul 0 a)) (HSMul.hSMul 0 (spectrum 𝕜 a))
    -/
  · simpa [ha, zero_smul_set] using (show {(0 : 𝕜)} = (0 : Set 𝕜) from rfl)
    /-
      🎉 no goals
    -/
    /-
      case inr
      𝕜 : Type u
      A : Type v
      inst✝³ : Field 𝕜
      inst✝² : Ring A
      inst✝¹ : Algebra 𝕜 A
      inst✝ : Nontrivial A
      k : 𝕜
      a : A
      ha : (spectrum 𝕜 a).Nonempty
      h : Ne k 0
      ⊢ Eq (spectrum 𝕜 (HSMul.hSMul k a)) (HSMul.hSMul k (spectrum 𝕜 a))
    -/
  · exact unit_smul_eq_smul a (Units.mk0 k h)
    /-
      🎉 no goals
    -/


theorem nonzero_mul_eq_swap_mul (a b : A) : σ (a * b) \ {0} = σ (b * a) \ {0} := by
  suffices h : ∀ x y : A, σ (x * y) \ {0} ⊆ σ (y * x) \ {0} from
    Set.eq_of_subset_of_subset (h a b) (h b a)
  /-
    𝕜 : Type u
    A : Type v
    inst✝² : Field 𝕜
    inst✝¹ : Ring A
    inst✝ : Algebra 𝕜 A
    a b : A
    ⊢ ∀ (x y : A), HasSubset.Subset (SDiff.sdiff (spectrum 𝕜 (HMul.hMul x y)) (Sin …
  -/
  rintro _ _ k ⟨k_mem, k_neq⟩
  /-
    case intro
    𝕜 : Type u
    A : Type v
    inst✝² : Field 𝕜
    inst✝¹ : Ring A
    inst✝ : Algebra 𝕜 A
    a b x✝ y✝ : A
    k : 𝕜
    k_mem : Membership.mem (spectrum 𝕜 (HMul.hMul x✝ y✝)) k
    k_neq : Not (Membership.mem (Singleton.singleton 0) k)
    ⊢ Membership.mem (SDiff.sdiff (spectrum 𝕜 (HMul.hMul y✝ x✝)) (Singleton.single …
  -/
  change ((Units.mk0 k k_neq) : 𝕜) ∈ _ at k_mem
  /-
    case intro
    𝕜 : Type u
    A : Type v
    inst✝² : Field 𝕜
    inst✝¹ : Ring A
    inst✝ : Algebra 𝕜 A
    a b x✝ y✝ : A
    k : 𝕜
    k_neq : Not (Membership.mem (Singleton.singleton 0) k)
    k_mem : Membership.mem (spectrum 𝕜 (HMul.hMul x✝ y✝)) ↑(Units.mk0 k k_neq)
    ⊢ Membership.mem (SDiff.sdiff (spectrum 𝕜 (HMul.hMul y✝ x✝)) (Singleton.single …
  -/
  exact ⟨unit_mem_mul_iff_mem_swap_mul.mp k_mem, k_neq⟩
  /-
    🎉 no goals
  -/


protected theorem map_inv (a : Aˣ) : (σ (a : A))⁻¹ = σ (↑a⁻¹ : A) := by
  /-
    𝕜 : Type u
    A : Type v
    inst✝² : Field 𝕜
    inst✝¹ : Ring A
    inst✝ : Algebra 𝕜 A
    a : Units A
    ⊢ Eq (Inv.inv (spectrum 𝕜 ↑a)) (spectrum 𝕜 ↑(Inv.inv a))
  -/
  refine Set.eq_of_subset_of_subset (fun k hk => ?_) fun k hk => ?_
    /-
      case refine_1
      𝕜 : Type u
      A : Type v
      inst✝² : Field 𝕜
      inst✝¹ : Ring A
      inst✝ : Algebra 𝕜 A
      a : Units A
      k : 𝕜
      hk : Membership.mem (Inv.inv (spectrum 𝕜 ↑a)) k
      ⊢ Membership.mem (spectrum 𝕜 ↑(Inv.inv a)) k
    -/
  · rw [Set.mem_inv] at hk
    /-
      case refine_1
      𝕜 : Type u
      A : Type v
      inst✝² : Field 𝕜
      inst✝¹ : Ring A
      inst✝ : Algebra 𝕜 A
      a : Units A
      k : 𝕜
      hk : Membership.mem (spectrum 𝕜 ↑a) (Inv.inv k)
      ⊢ Membership.mem (spectrum 𝕜 ↑(Inv.inv a)) k
    -/
    have : k ≠ 0 := by simpa only [inv_inv] using inv_ne_zero (ne_zero_of_mem_of_unit hk)
    /-
      case refine_1
      𝕜 : Type u
      A : Type v
      inst✝² : Field 𝕜
      inst✝¹ : Ring A
      inst✝ : Algebra 𝕜 A
      a : Units A
      k : 𝕜
      hk : Membership.mem (spectrum 𝕜 ↑a) (Inv.inv k)
      this : Ne k 0
      ⊢ Membership.mem (spectrum 𝕜 ↑(Inv.inv a)) k
    -/
    lift k to 𝕜ˣ using isUnit_iff_ne_zero.mpr this
    /-
      case refine_1.intro
      𝕜 : Type u
      A : Type v
      inst✝² : Field 𝕜
      inst✝¹ : Ring A
      inst✝ : Algebra 𝕜 A
      a : Units A
      k : Units 𝕜
      hk : Membership.mem (spectrum 𝕜 ↑a) (Inv.inv ↑k)
      this : Ne (↑k) 0
      ⊢ Membership.mem (spectrum 𝕜 ↑(Inv.inv a)) ↑k
    -/
    rw [← Units.val_inv_eq_inv_val k] at hk
    /-
      case refine_1.intro
      𝕜 : Type u
      A : Type v
      inst✝² : Field 𝕜
      inst✝¹ : Ring A
      inst✝ : Algebra 𝕜 A
      a : Units A
      k : Units 𝕜
      hk : Membership.mem (spectrum 𝕜 ↑a) ↑(Inv.inv k)
      this : Ne (↑k) 0
      ⊢ Membership.mem (spectrum 𝕜 ↑(Inv.inv a)) ↑k
    -/
    exact inv_mem_iff.mp hk
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      𝕜 : Type u
      A : Type v
      inst✝² : Field 𝕜
      inst✝¹ : Ring A
      inst✝ : Algebra 𝕜 A
      a : Units A
      k : 𝕜
      hk : Membership.mem (spectrum 𝕜 ↑(Inv.inv a)) k
      ⊢ Membership.mem (Inv.inv (spectrum 𝕜 ↑a)) k
    -/
  · lift k to 𝕜ˣ using isUnit_iff_ne_zero.mpr (ne_zero_of_mem_of_unit hk)
    /-
      case refine_2.intro
      𝕜 : Type u
      A : Type v
      inst✝² : Field 𝕜
      inst✝¹ : Ring A
      inst✝ : Algebra 𝕜 A
      a : Units A
      k : Units 𝕜
      hk : Membership.mem (spectrum 𝕜 ↑(Inv.inv a)) ↑k
      ⊢ Membership.mem (Inv.inv (spectrum 𝕜 ↑a)) ↑k
    -/
    simpa only [Units.val_inv_eq_inv_val] using inv_mem_iff.mp hk
    /-
      🎉 no goals
    -/


theorem mem_resolventSet_apply (φ : F) {a : A} {r : R} (h : r ∈ resolventSet R a) :
    r ∈ resolventSet R ((φ : A → B) a) := by
  /-
    F : Type u_1
    R : Type u_2
    A : Type u_3
    B : Type u_4
    inst✝⁶ : CommSemiring R
    inst✝⁵ : Ring A
    inst✝⁴ : Algebra R A
    inst✝³ : Ring B
    inst✝² : Algebra R B
    inst✝¹ : FunLike F A B
    inst✝ : AlgHomClass F R A B
    φ : F
    a : A
    r : R
    h : Membership.mem (resolventSet R a) r
    ⊢ Membership.mem (resolventSet R (φ a)) r
  -/
  simpa only [map_sub, AlgHomClass.commutes] using h.map φ
  /-
    🎉 no goals
  -/


theorem spectrum_apply_subset (φ : F) (a : A) : σ ((φ : A → B) a) ⊆ σ a := fun _ =>
  mt (mem_resolventSet_apply φ)


theorem apply_mem_spectrum [Nontrivial R] (φ : F) (a : A) : φ a ∈ σ a := by
  have h : ↑ₐ (φ a) - a ∈ RingHom.ker (φ : A →+* R) := by
    simp only [RingHom.mem_ker, map_sub, RingHom.coe_coe, AlgHomClass.commutes,
      Algebra.id.map_eq_id, RingHom.id_apply, sub_self]
  simp only [spectrum.mem_iff, ← mem_nonunits_iff,
    coe_subset_nonunits (RingHom.ker_ne_top (φ : A →+* R)) h]


@[simp]
theorem AlgEquiv.spectrum_eq {F R A B : Type*} [CommSemiring R] [Ring A] [Ring B] [Algebra R A]
    [Algebra R B] [EquivLike F A B] [AlgEquivClass F R A B] (f : F) (a : A) :
    spectrum R (f a) = spectrum R a :=
  Set.Subset.antisymm (AlgHom.spectrum_apply_subset _ _) <| by
    simpa only [AlgEquiv.coe_algHom, AlgEquiv.coe_coe_symm_apply_coe_apply] using
      AlgHom.spectrum_apply_subset (f : A ≃ₐ[R] B).symm (f a)


/-- Conjugation by a unit preserves the spectrum, inverse on right. -/
@[simp]
lemma spectrum.units_conjugate {a : A} {u : Aˣ} :
    spectrum R (u * a * u⁻¹) = spectrum R a := by
  suffices ∀ (b : A) (v : Aˣ), spectrum R (v * b * v⁻¹) ⊆ spectrum R b by
    refine le_antisymm (this a u) ?_
    apply le_of_eq_of_le ?_ <| this (u * a * u⁻¹) u⁻¹
    simp [mul_assoc]
  /-
    R : Type u_1
    A : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : Ring A
    inst✝ : Algebra R A
    a : A
    u : Units A
    ⊢ ∀ (b : A) (v : Units A), HasSubset.Subset (spectrum R (HMul.hMul (HMul.hMul  …
  -/
  intro a u μ hμ
  /-
    R : Type u_1
    A : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : Ring A
    inst✝ : Algebra R A
    a✝ : A
    u✝ : Units A
    a : A
    u : Units A
    μ : R
    hμ : Membership.mem (spectrum R (HMul.hMul (HMul.hMul (↑u) a) ↑(Inv.inv u))) μ
    ⊢ Membership.mem (spectrum R a) μ
  -/
  rw [spectrum.mem_iff] at hμ ⊢
  /-
    R : Type u_1
    A : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : Ring A
    inst✝ : Algebra R A
    a✝ : A
    u✝ : Units A
    a : A
    u : Units A
    μ : R
    hμ : Not (IsUnit (HSub.hSub ((algebraMap R A) μ) (HMul.hMul (HMul.hMul (↑u) a) …
    ⊢ Not (IsUnit (HSub.hSub ((algebraMap R A) μ) a))
  -/
  contrapose! hμ
  /-
    R : Type u_1
    A : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : Ring A
    inst✝ : Algebra R A
    a✝ : A
    u✝ : Units A
    a : A
    u : Units A
    μ : R
    hμ : IsUnit (HSub.hSub ((algebraMap R A) μ) a)
    ⊢ IsUnit (HSub.hSub ((algebraMap R A) μ) (HMul.hMul (HMul.hMul (↑u) a) ↑(Inv.i …
  -/
  simpa [mul_sub, sub_mul, Algebra.right_comm] using u.isUnit.mul hμ |>.mul u⁻¹.isUnit
  /-
    🎉 no goals
  -/


/-- Conjugation by a unit preserves the spectrum, inverse on left. -/
@[simp]
lemma spectrum.units_conjugate' {a : A} {u : Aˣ} :
    spectrum R (u⁻¹ * a * u) = spectrum R a := by
  /-
    R : Type u_1
    A : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : Ring A
    inst✝ : Algebra R A
    a : A
    u : Units A
    ⊢ Eq (spectrum R (HMul.hMul (HMul.hMul (↑(Inv.inv u)) a) ↑u)) (spectrum R a)
  -/
  simpa using spectrum.units_conjugate (u := u⁻¹)
  /-
    🎉 no goals
  -/


