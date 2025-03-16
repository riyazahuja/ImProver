theorem SetLike.algebraMap_mem_graded [Zero ι] [CommSemiring S] [Semiring R] [Algebra S R]
    (A : ι → Submodule S R) [SetLike.GradedOne A] (s : S) : algebraMap S R s ∈ A 0 := by
  /-
    ι : Type u_1
    S : Type u_3
    R : Type u_4
    inst✝⁴ : Zero ι
    inst✝³ : CommSemiring S
    inst✝² : Semiring R
    inst✝¹ : Algebra S R
    A : ι → Submodule S R
    inst✝ : SetLike.GradedOne A
    s : S
    ⊢ Membership.mem (A 0) ((algebraMap S R) s)
  -/
  rw [Algebra.algebraMap_eq_smul_one]
  /-
    ι : Type u_1
    S : Type u_3
    R : Type u_4
    inst✝⁴ : Zero ι
    inst✝³ : CommSemiring S
    inst✝² : Semiring R
    inst✝¹ : Algebra S R
    A : ι → Submodule S R
    inst✝ : SetLike.GradedOne A
    s : S
    ⊢ Membership.mem (A 0) (HSMul.hSMul s 1)
  -/
  exact (A 0).smul_mem s <| SetLike.one_mem_graded _
  /-
    🎉 no goals
  -/


theorem SetLike.natCast_mem_graded [Zero ι] [AddMonoidWithOne R] [SetLike σ R]
    [AddSubmonoidClass σ R] (A : ι → σ) [SetLike.GradedOne A] (n : ℕ) : (n : R) ∈ A 0 := by
  induction n with
  | zero =>
    rw [Nat.cast_zero]
    exact zero_mem (A 0)
  | succ _ n_ih =>
    rw [Nat.cast_succ]
    exact add_mem n_ih (SetLike.one_mem_graded _)


@[deprecated (since := "2024-04-17")]
alias SetLike.nat_cast_mem_graded := SetLike.natCast_mem_graded


theorem SetLike.intCast_mem_graded [Zero ι] [AddGroupWithOne R] [SetLike σ R]
    [AddSubgroupClass σ R] (A : ι → σ) [SetLike.GradedOne A] (z : ℤ) : (z : R) ∈ A 0 := by
  /-
    ι : Type u_1
    σ : Type u_2
    R : Type u_4
    inst✝⁴ : Zero ι
    inst✝³ : AddGroupWithOne R
    inst✝² : SetLike σ R
    inst✝¹ : AddSubgroupClass σ R
    A : ι → σ
    inst✝ : SetLike.GradedOne A
    z : Int
    ⊢ Membership.mem (A 0) ↑z
  -/
  induction z
    /-
      case ofNat
      ι : Type u_1
      σ : Type u_2
      R : Type u_4
      inst✝⁴ : Zero ι
      inst✝³ : AddGroupWithOne R
      inst✝² : SetLike σ R
      inst✝¹ : AddSubgroupClass σ R
      A : ι → σ
      inst✝ : SetLike.GradedOne A
      a✝ : Nat
      ⊢ Membership.mem (A 0) ↑(Int.ofNat a✝)
    -/
  · rw [Int.ofNat_eq_coe, Int.cast_natCast]
    /-
      case ofNat
      ι : Type u_1
      σ : Type u_2
      R : Type u_4
      inst✝⁴ : Zero ι
      inst✝³ : AddGroupWithOne R
      inst✝² : SetLike σ R
      inst✝¹ : AddSubgroupClass σ R
      A : ι → σ
      inst✝ : SetLike.GradedOne A
      a✝ : Nat
      ⊢ Membership.mem (A 0) ↑a✝
    -/
    exact SetLike.natCast_mem_graded _ _
    /-
      🎉 no goals
    -/
    /-
      case negSucc
      ι : Type u_1
      σ : Type u_2
      R : Type u_4
      inst✝⁴ : Zero ι
      inst✝³ : AddGroupWithOne R
      inst✝² : SetLike σ R
      inst✝¹ : AddSubgroupClass σ R
      A : ι → σ
      inst✝ : SetLike.GradedOne A
      a✝ : Nat
      ⊢ Membership.mem (A 0) ↑(Int.negSucc a✝)
    -/
  · rw [Int.cast_negSucc]
    /-
      case negSucc
      ι : Type u_1
      σ : Type u_2
      R : Type u_4
      inst✝⁴ : Zero ι
      inst✝³ : AddGroupWithOne R
      inst✝² : SetLike σ R
      inst✝¹ : AddSubgroupClass σ R
      A : ι → σ
      inst✝ : SetLike.GradedOne A
      a✝ : Nat
      ⊢ Membership.mem (A 0) (Neg.neg ↑(HAdd.hAdd a✝ 1))
    -/
    exact neg_mem (SetLike.natCast_mem_graded _ _)
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-04-17")]
alias SetLike.int_cast_mem_graded := SetLike.intCast_mem_graded


/-- Build a `DirectSum.GNonUnitalNonAssocSemiring` instance for a collection of additive
submonoids. -/
instance gnonUnitalNonAssocSemiring [Add ι] [NonUnitalNonAssocSemiring R] [SetLike σ R]
    [AddSubmonoidClass σ R] (A : ι → σ) [SetLike.GradedMul A] :
    DirectSum.GNonUnitalNonAssocSemiring fun i => A i :=
  { SetLike.gMul A with
    mul_zero := fun _ => Subtype.ext (mul_zero _)
    zero_mul := fun _ => Subtype.ext (zero_mul _)
    mul_add := fun _ _ _ => Subtype.ext (mul_add _ _ _)
    add_mul := fun _ _ _ => Subtype.ext (add_mul _ _ _) }


/-- Build a `DirectSum.GSemiring` instance for a collection of additive submonoids. -/
instance gsemiring [AddMonoid ι] [Semiring R] [SetLike σ R] [AddSubmonoidClass σ R] (A : ι → σ)
    [SetLike.GradedMonoid A] : DirectSum.GSemiring fun i => A i :=
  { SetLike.gMonoid A with
    mul_zero := fun _ => Subtype.ext (mul_zero _)
    zero_mul := fun _ => Subtype.ext (zero_mul _)
    mul_add := fun _ _ _ => Subtype.ext (mul_add _ _ _)
    add_mul := fun _ _ _ => Subtype.ext (add_mul _ _ _)
    natCast := fun n => ⟨n, SetLike.natCast_mem_graded _ _⟩
    natCast_zero := Subtype.ext Nat.cast_zero
    natCast_succ := fun n => Subtype.ext (Nat.cast_succ n) }


/-- Build a `DirectSum.GCommSemiring` instance for a collection of additive submonoids. -/
instance gcommSemiring [AddCommMonoid ι] [CommSemiring R] [SetLike σ R] [AddSubmonoidClass σ R]
    (A : ι → σ) [SetLike.GradedMonoid A] : DirectSum.GCommSemiring fun i => A i :=
  { SetLike.gCommMonoid A, SetLike.gsemiring A with }


/-- Build a `DirectSum.GRing` instance for a collection of additive subgroups. -/
instance gring [AddMonoid ι] [Ring R] [SetLike σ R] [AddSubgroupClass σ R] (A : ι → σ)
    [SetLike.GradedMonoid A] : DirectSum.GRing fun i => A i :=
  { SetLike.gsemiring A with
    intCast := fun z => ⟨z, SetLike.intCast_mem_graded _ _⟩
    intCast_ofNat := fun _n => Subtype.ext <| Int.cast_natCast _
    intCast_negSucc_ofNat := fun n => Subtype.ext <| Int.cast_negSucc n }


/-- Build a `DirectSum.GCommRing` instance for a collection of additive submonoids. -/
instance gcommRing [AddCommMonoid ι] [CommRing R] [SetLike σ R] [AddSubgroupClass σ R] (A : ι → σ)
    [SetLike.GradedMonoid A] : DirectSum.GCommRing fun i => A i :=
  { SetLike.gCommMonoid A, SetLike.gring A with }


/-- The canonical ring isomorphism between `⨁ i, A i` and `R`-/
def coeRingHom [AddMonoid ι] [SetLike.GradedMonoid A] : (⨁ i, A i) →+* R :=
  DirectSum.toSemiring (fun i => AddSubmonoidClass.subtype (A i)) rfl fun _ _ => rfl


/-- The canonical ring isomorphism between `⨁ i, A i` and `R`-/
@[simp]
theorem coeRingHom_of [AddMonoid ι] [SetLike.GradedMonoid A] (i : ι) (x : A i) :
    (coeRingHom A : _ →+* R) (of (fun i => A i) i x) = x :=
  DirectSum.toSemiring_of _ _ _ _ _


theorem coe_mul_apply [AddMonoid ι] [SetLike.GradedMonoid A]
    [∀ (i : ι) (x : A i), Decidable (x ≠ 0)] (r r' : ⨁ i, A i) (n : ι) :
    ((r * r') n : R) =
      ∑ ij ∈ (r.support ×ˢ r'.support).filter (fun ij : ι × ι => ij.1 + ij.2 = n),
        (r ij.1 * r' ij.2 : R) := by
  /-
    ι : Type u_1
    σ : Type u_2
    R : Type u_4
    inst✝⁶ : DecidableEq ι
    inst✝⁵ : Semiring R
    inst✝⁴ : SetLike σ R
    inst✝³ : AddSubmonoidClass σ R
    A : ι → σ
    inst✝² : AddMonoid ι
    inst✝¹ : SetLike.GradedMonoid A
    inst✝ : (i : ι) → (x : Subtype fun x => Membership.mem (A i) x) → Decidable (N …
    r r' : DirectSum ι fun i => Subtype fun x => Membership.mem (A i) x
    n : ι
    ⊢ Eq (↑((HMul.hMul r r') n)) ((Finset.filter (fun ij => Eq (HAdd.hAdd ij.1 ij. …
  -/
  rw [mul_eq_sum_support_ghas_mul, DFinsupp.finset_sum_apply, AddSubmonoidClass.coe_finset_sum]
  /-
    ι : Type u_1
    σ : Type u_2
    R : Type u_4
    inst✝⁶ : DecidableEq ι
    inst✝⁵ : Semiring R
    inst✝⁴ : SetLike σ R
    inst✝³ : AddSubmonoidClass σ R
    A : ι → σ
    inst✝² : AddMonoid ι
    inst✝¹ : SetLike.GradedMonoid A
    inst✝ : (i : ι) → (x : Subtype fun x => Membership.mem (A i) x) → Decidable (N …
    r r' : DirectSum ι fun i => Subtype fun x => Membership.mem (A i) x
    n : ι
    ⊢ Eq ((SProd.sprod (DFinsupp.support r) (DFinsupp.support r')).sum fun i => ↑( …
  -/
  simp_rw [coe_of_apply, apply_ite, ZeroMemClass.coe_zero, ← Finset.sum_filter, SetLike.coe_gMul]
  /-
    🎉 no goals
  -/


theorem coe_mul_apply_eq_dfinsupp_sum [AddMonoid ι] [SetLike.GradedMonoid A]
    [∀ (i : ι) (x : A i), Decidable (x ≠ 0)] (r r' : ⨁ i, A i) (n : ι) :
    ((r * r') n : R) = r.sum fun i ri => r'.sum fun j rj => if i + j = n then (ri * rj : R)
      else 0 := by
  /-
    ι : Type u_1
    σ : Type u_2
    R : Type u_4
    inst✝⁶ : DecidableEq ι
    inst✝⁵ : Semiring R
    inst✝⁴ : SetLike σ R
    inst✝³ : AddSubmonoidClass σ R
    A : ι → σ
    inst✝² : AddMonoid ι
    inst✝¹ : SetLike.GradedMonoid A
    inst✝ : (i : ι) → (x : Subtype fun x => Membership.mem (A i) x) → Decidable (N …
    r r' : DirectSum ι fun i => Subtype fun x => Membership.mem (A i) x
    n : ι
    ⊢ Eq (↑((HMul.hMul r r') n)) (DFinsupp.sum r fun i ri => DFinsupp.sum r' fun j …
  -/
  rw [mul_eq_dfinsupp_sum]
  /-
    ι : Type u_1
    σ : Type u_2
    R : Type u_4
    inst✝⁶ : DecidableEq ι
    inst✝⁵ : Semiring R
    inst✝⁴ : SetLike σ R
    inst✝³ : AddSubmonoidClass σ R
    A : ι → σ
    inst✝² : AddMonoid ι
    inst✝¹ : SetLike.GradedMonoid A
    inst✝ : (i : ι) → (x : Subtype fun x => Membership.mem (A i) x) → Decidable (N …
    r r' : DirectSum ι fun i => Subtype fun x => Membership.mem (A i) x
    n : ι
    ⊢ Eq (↑((DFinsupp.sum r fun x ai => DFinsupp.sum r' fun x_1 aj => (DirectSum.o …
  -/
  iterate 2 rw [DFinsupp.sum_apply, DFinsupp.sum, AddSubmonoidClass.coe_finset_sum]; congr; ext
  /-
    case e_f.h.e_f.h
    ι : Type u_1
    σ : Type u_2
    R : Type u_4
    inst✝⁶ : DecidableEq ι
    inst✝⁵ : Semiring R
    inst✝⁴ : SetLike σ R
    inst✝³ : AddSubmonoidClass σ R
    A : ι → σ
    inst✝² : AddMonoid ι
    inst✝¹ : SetLike.GradedMonoid A
    inst✝ : (i : ι) → (x : Subtype fun x => Membership.mem (A i) x) → Decidable (N …
    r r' : DirectSum ι fun i => Subtype fun x => Membership.mem (A i) x
    n x✝¹ x✝ : ι
    ⊢ Eq (↑(((DirectSum.of (fun i => Subtype fun x => Membership.mem (A i) x) (HAd …
  -/
  dsimp only
  /-
    case e_f.h.e_f.h
    ι : Type u_1
    σ : Type u_2
    R : Type u_4
    inst✝⁶ : DecidableEq ι
    inst✝⁵ : Semiring R
    inst✝⁴ : SetLike σ R
    inst✝³ : AddSubmonoidClass σ R
    A : ι → σ
    inst✝² : AddMonoid ι
    inst✝¹ : SetLike.GradedMonoid A
    inst✝ : (i : ι) → (x : Subtype fun x => Membership.mem (A i) x) → Decidable (N …
    r r' : DirectSum ι fun i => Subtype fun x => Membership.mem (A i) x
    n x✝¹ x✝ : ι
    ⊢ Eq (↑(((DirectSum.of (fun i => Subtype fun x => Membership.mem (A i) x) (HAd …
  -/
  split_ifs with h
    /-
      case pos
      ι : Type u_1
      σ : Type u_2
      R : Type u_4
      inst✝⁶ : DecidableEq ι
      inst✝⁵ : Semiring R
      inst✝⁴ : SetLike σ R
      inst✝³ : AddSubmonoidClass σ R
      A : ι → σ
      inst✝² : AddMonoid ι
      inst✝¹ : SetLike.GradedMonoid A
      inst✝ : (i : ι) → (x : Subtype fun x => Membership.mem (A i) x) → Decidable (N …
      r r' : DirectSum ι fun i => Subtype fun x => Membership.mem (A i) x
      n x✝¹ x✝ : ι
      h : Eq (HAdd.hAdd x✝¹ x✝) n
      ⊢ Eq (↑(((DirectSum.of (fun i => Subtype fun x => Membership.mem (A i) x) (HAd …
    -/
  · subst h
    /-
      case pos
      ι : Type u_1
      σ : Type u_2
      R : Type u_4
      inst✝⁶ : DecidableEq ι
      inst✝⁵ : Semiring R
      inst✝⁴ : SetLike σ R
      inst✝³ : AddSubmonoidClass σ R
      A : ι → σ
      inst✝² : AddMonoid ι
      inst✝¹ : SetLike.GradedMonoid A
      inst✝ : (i : ι) → (x : Subtype fun x => Membership.mem (A i) x) → Decidable (N …
      r r' : DirectSum ι fun i => Subtype fun x => Membership.mem (A i) x
      x✝¹ x✝ : ι
      ⊢ Eq (↑(((DirectSum.of (fun i => Subtype fun x => Membership.mem (A i) x) (HAd …
    -/
    rw [of_eq_same]
    /-
      case pos
      ι : Type u_1
      σ : Type u_2
      R : Type u_4
      inst✝⁶ : DecidableEq ι
      inst✝⁵ : Semiring R
      inst✝⁴ : SetLike σ R
      inst✝³ : AddSubmonoidClass σ R
      A : ι → σ
      inst✝² : AddMonoid ι
      inst✝¹ : SetLike.GradedMonoid A
      inst✝ : (i : ι) → (x : Subtype fun x => Membership.mem (A i) x) → Decidable (N …
      r r' : DirectSum ι fun i => Subtype fun x => Membership.mem (A i) x
      x✝¹ x✝ : ι
      ⊢ Eq (↑(GradedMonoid.GMul.mul (r x✝¹) (r' x✝))) (HMul.hMul ↑(r x✝¹) ↑(r' x✝))
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case neg
      ι : Type u_1
      σ : Type u_2
      R : Type u_4
      inst✝⁶ : DecidableEq ι
      inst✝⁵ : Semiring R
      inst✝⁴ : SetLike σ R
      inst✝³ : AddSubmonoidClass σ R
      A : ι → σ
      inst✝² : AddMonoid ι
      inst✝¹ : SetLike.GradedMonoid A
      inst✝ : (i : ι) → (x : Subtype fun x => Membership.mem (A i) x) → Decidable (N …
      r r' : DirectSum ι fun i => Subtype fun x => Membership.mem (A i) x
      n x✝¹ x✝ : ι
      h : Not (Eq (HAdd.hAdd x✝¹ x✝) n)
      ⊢ Eq (↑(((DirectSum.of (fun i => Subtype fun x => Membership.mem (A i) x) (HAd …
    -/
  · rw [of_eq_of_ne _ _ _ h]
    /-
      case neg
      ι : Type u_1
      σ : Type u_2
      R : Type u_4
      inst✝⁶ : DecidableEq ι
      inst✝⁵ : Semiring R
      inst✝⁴ : SetLike σ R
      inst✝³ : AddSubmonoidClass σ R
      A : ι → σ
      inst✝² : AddMonoid ι
      inst✝¹ : SetLike.GradedMonoid A
      inst✝ : (i : ι) → (x : Subtype fun x => Membership.mem (A i) x) → Decidable (N …
      r r' : DirectSum ι fun i => Subtype fun x => Membership.mem (A i) x
      n x✝¹ x✝ : ι
      h : Not (Eq (HAdd.hAdd x✝¹ x✝) n)
      ⊢ Eq (↑0) 0
    -/
    rfl
    /-
      🎉 no goals
    -/


theorem coe_of_mul_apply_aux [AddMonoid ι] [SetLike.GradedMonoid A] {i : ι} (r : A i)
    (r' : ⨁ i, A i) {j n : ι} (H : ∀ x : ι, i + x = n ↔ x = j) :
    ((of (fun i => A i) i r * r') n : R) = r * r' j := by
  classical
    rw [coe_mul_apply_eq_dfinsupp_sum]
    apply (DFinsupp.sum_single_index _).trans
    swap
    · simp_rw [ZeroMemClass.coe_zero, zero_mul, ite_self]
      exact DFinsupp.sum_zero
    simp_rw [DFinsupp.sum, H, Finset.sum_ite_eq']
    split_ifs with h
    · rfl
    rw [DFinsupp.not_mem_support_iff.mp h, ZeroMemClass.coe_zero, mul_zero]


theorem coe_mul_of_apply_aux [AddMonoid ι] [SetLike.GradedMonoid A] (r : ⨁ i, A i) {i : ι}
    (r' : A i) {j n : ι} (H : ∀ x : ι, x + i = n ↔ x = j) :
    ((r * of (fun i => A i) i r') n : R) = r j * r' := by
  classical
    rw [coe_mul_apply_eq_dfinsupp_sum, DFinsupp.sum_comm]
    apply (DFinsupp.sum_single_index _).trans
    swap
    · simp_rw [ZeroMemClass.coe_zero, mul_zero, ite_self]
      exact DFinsupp.sum_zero
    simp_rw [DFinsupp.sum, H, Finset.sum_ite_eq']
    split_ifs with h
    · rfl
    rw [DFinsupp.not_mem_support_iff.mp h, ZeroMemClass.coe_zero, zero_mul]


theorem coe_of_mul_apply_add [AddLeftCancelMonoid ι] [SetLike.GradedMonoid A] {i : ι} (r : A i)
    (r' : ⨁ i, A i) (j : ι) : ((of (fun i => A i) i r * r') (i + j) : R) = r * r' j :=
  coe_of_mul_apply_aux _ _ _ fun _x => ⟨fun h => add_left_cancel h, fun h => h ▸ rfl⟩


theorem coe_mul_of_apply_add [AddRightCancelMonoid ι] [SetLike.GradedMonoid A] (r : ⨁ i, A i)
    {i : ι} (r' : A i) (j : ι) : ((r * of (fun i => A i) i r') (j + i) : R) = r j * r' :=
  coe_mul_of_apply_aux _ _ _ fun _x => ⟨fun h => add_right_cancel h, fun h => h ▸ rfl⟩


theorem coe_of_mul_apply_of_not_le {i : ι} (r : A i) (r' : ⨁ i, A i) (n : ι) (h : ¬i ≤ n) :
    ((of (fun i => A i) i r * r') n : R) = 0 := by
  classical
    rw [coe_mul_apply_eq_dfinsupp_sum]
    apply (DFinsupp.sum_single_index _).trans
    swap
    · simp_rw [ZeroMemClass.coe_zero, zero_mul, ite_self]
      exact DFinsupp.sum_zero
    · rw [DFinsupp.sum, Finset.sum_ite_of_false, Finset.sum_const_zero]
      exact fun x _ H => h ((self_le_add_right i x).trans_eq H)


theorem coe_mul_of_apply_of_not_le (r : ⨁ i, A i) {i : ι} (r' : A i) (n : ι) (h : ¬i ≤ n) :
    ((r * of (fun i => A i) i r') n : R) = 0 := by
  classical
    rw [coe_mul_apply_eq_dfinsupp_sum, DFinsupp.sum_comm]
    apply (DFinsupp.sum_single_index _).trans
    swap
    · simp_rw [ZeroMemClass.coe_zero, mul_zero, ite_self]
      exact DFinsupp.sum_zero
    · rw [DFinsupp.sum, Finset.sum_ite_of_false, Finset.sum_const_zero]
      exact fun x _ H => h ((self_le_add_left i x).trans_eq H)


theorem coe_mul_of_apply_of_le (r : ⨁ i, A i) {i : ι} (r' : A i) (n : ι) (h : i ≤ n) :
    ((r * of (fun i => A i) i r') n : R) = r (n - i) * r' :=
  coe_mul_of_apply_aux _ _ _ fun _x => (eq_tsub_iff_add_eq_of_le h).symm


theorem coe_of_mul_apply_of_le {i : ι} (r : A i) (r' : ⨁ i, A i) (n : ι) (h : i ≤ n) :
    ((of (fun i => A i) i r * r') n : R) = r * r' (n - i) :=
                                         /-
                                           ι : Type u_1
                                           σ : Type u_2
                                           R : Type u_4
                                           inst✝⁸ : DecidableEq ι
                                           inst✝⁷ : Semiring R
                                           inst✝⁶ : SetLike σ R
                                           inst✝⁵ : AddSubmonoidClass σ R
                                           A : ι → σ
                                           inst✝⁴ : CanonicallyOrderedAddCommMonoid ι
                                           inst✝³ : SetLike.GradedMonoid A
                                           inst✝² : Sub ι
                                           inst✝¹ : OrderedSub ι
                                           inst✝ : AddLeftReflectLE ι
                                           i : ι
                                           r : Subtype fun x => Membership.mem (A i) x
                                           r' : DirectSum ι fun i => Subtype fun x => Membership.mem (A i) x
                                           n : ι
                                           h : LE.le i n
                                           x : ι
                                           ⊢ Iff (Eq (HAdd.hAdd i x) n) (Eq x (HSub.hSub n i))
                                         -/
  coe_of_mul_apply_aux _ _ _ fun x => by rw [eq_tsub_iff_add_eq_of_le h, add_comm]
                                         /-
                                           🎉 no goals
                                         -/


theorem coe_mul_of_apply (r : ⨁ i, A i) {i : ι} (r' : A i) (n : ι) [Decidable (i ≤ n)] :
    ((r * of (fun i => A i) i r') n : R) = if i ≤ n then (r (n - i) : R) * r' else 0 := by
  /-
    ι : Type u_1
    σ : Type u_2
    R : Type u_4
    inst✝⁹ : DecidableEq ι
    inst✝⁸ : Semiring R
    inst✝⁷ : SetLike σ R
    inst✝⁶ : AddSubmonoidClass σ R
    A : ι → σ
    inst✝⁵ : CanonicallyOrderedAddCommMonoid ι
    inst✝⁴ : SetLike.GradedMonoid A
    inst✝³ : Sub ι
    inst✝² : OrderedSub ι
    inst✝¹ : AddLeftReflectLE ι
    r : DirectSum ι fun i => Subtype fun x => Membership.mem (A i) x
    i : ι
    r' : Subtype fun x => Membership.mem (A i) x
    n : ι
    inst✝ : Decidable (LE.le i n)
    ⊢ Eq (↑((HMul.hMul r ((DirectSum.of (fun i => Subtype fun x => Membership.mem  …
  -/
  split_ifs with h
  /-
    case pos
    ι : Type u_1
    σ : Type u_2
    R : Type u_4
    inst✝⁹ : DecidableEq ι
    inst✝⁸ : Semiring R
    inst✝⁷ : SetLike σ R
    inst✝⁶ : AddSubmonoidClass σ R
    A : ι → σ
    inst✝⁵ : CanonicallyOrderedAddCommMonoid ι
    inst✝⁴ : SetLike.GradedMonoid A
    inst✝³ : Sub ι
    inst✝² : OrderedSub ι
    inst✝¹ : AddLeftReflectLE ι
    r : DirectSum ι fun i => Subtype fun x => Membership.mem (A i) x
    i : ι
    r' : Subtype fun x => Membership.mem (A i) x
    n : ι
    inst✝ : Decidable (LE.le i n)
    h : LE.le i n
    ⊢ Eq (↑((HMul.hMul r ((DirectSum.of (fun i => Subtype fun x => Membership.mem  …
  -/
  exacts [coe_mul_of_apply_of_le _ _ _ n h, coe_mul_of_apply_of_not_le _ _ _ n h]
  /-
    🎉 no goals
  -/


theorem coe_of_mul_apply {i : ι} (r : A i) (r' : ⨁ i, A i) (n : ι) [Decidable (i ≤ n)] :
    ((of (fun i => A i) i r * r') n : R) = if i ≤ n then (r * r' (n - i) : R) else 0 := by
  /-
    ι : Type u_1
    σ : Type u_2
    R : Type u_4
    inst✝⁹ : DecidableEq ι
    inst✝⁸ : Semiring R
    inst✝⁷ : SetLike σ R
    inst✝⁶ : AddSubmonoidClass σ R
    A : ι → σ
    inst✝⁵ : CanonicallyOrderedAddCommMonoid ι
    inst✝⁴ : SetLike.GradedMonoid A
    inst✝³ : Sub ι
    inst✝² : OrderedSub ι
    inst✝¹ : AddLeftReflectLE ι
    i : ι
    r : Subtype fun x => Membership.mem (A i) x
    r' : DirectSum ι fun i => Subtype fun x => Membership.mem (A i) x
    n : ι
    inst✝ : Decidable (LE.le i n)
    ⊢ Eq (↑((HMul.hMul ((DirectSum.of (fun i => Subtype fun x => Membership.mem (A …
  -/
  split_ifs with h
  /-
    case pos
    ι : Type u_1
    σ : Type u_2
    R : Type u_4
    inst✝⁹ : DecidableEq ι
    inst✝⁸ : Semiring R
    inst✝⁷ : SetLike σ R
    inst✝⁶ : AddSubmonoidClass σ R
    A : ι → σ
    inst✝⁵ : CanonicallyOrderedAddCommMonoid ι
    inst✝⁴ : SetLike.GradedMonoid A
    inst✝³ : Sub ι
    inst✝² : OrderedSub ι
    inst✝¹ : AddLeftReflectLE ι
    i : ι
    r : Subtype fun x => Membership.mem (A i) x
    r' : DirectSum ι fun i => Subtype fun x => Membership.mem (A i) x
    n : ι
    inst✝ : Decidable (LE.le i n)
    h : LE.le i n
    ⊢ Eq (↑((HMul.hMul ((DirectSum.of (fun i => Subtype fun x => Membership.mem (A …
  -/
  exacts [coe_of_mul_apply_of_le _ _ _ n h, coe_of_mul_apply_of_not_le _ _ _ n h]
  /-
    🎉 no goals
  -/


/-- Build a `DirectSum.GAlgebra` instance for a collection of `Submodule`s. -/
instance galgebra [AddMonoid ι] [CommSemiring S] [Semiring R] [Algebra S R] (A : ι → Submodule S R)
    [SetLike.GradedMonoid A] : DirectSum.GAlgebra S fun i => A i where
  toFun :=
    ((Algebra.linearMap S R).codRestrict (A 0) <| SetLike.algebraMap_mem_graded A).toAddMonoidHom
  map_one := Subtype.ext <| (algebraMap S R).map_one
  map_mul _x _y := Sigma.subtype_ext (add_zero 0).symm <| (algebraMap S R).map_mul _ _
  commutes := fun _r ⟨i, _xi⟩ =>
    Sigma.subtype_ext ((zero_add i).trans (add_zero i).symm) <| Algebra.commutes _ _
  smul_def := fun _r ⟨i, _xi⟩ => Sigma.subtype_ext (zero_add i).symm <| Algebra.smul_def _ _


@[simp]
theorem setLike.coe_galgebra_toFun {ι} [AddMonoid ι] [CommSemiring S] [Semiring R] [Algebra S R]
    (A : ι → Submodule S R) [SetLike.GradedMonoid A] (s : S) :
    (DirectSum.GAlgebra.toFun (A := fun i => A i) s) = (algebraMap S R s : R) :=
  rfl


/-- A direct sum of powers of a submodule of an algebra has a multiplicative structure. -/
instance nat_power_gradedMonoid [CommSemiring S] [Semiring R] [Algebra S R] (p : Submodule S R) :
    SetLike.GradedMonoid fun i : ℕ => p ^ i where
  one_mem := by
    /-
      ι : Type u_1
      σ : Type u_2
      S : Type u_3
      R : Type u_4
      inst✝³ : DecidableEq ι
      inst✝² : CommSemiring S
      inst✝¹ : Semiring R
      inst✝ : Algebra S R
      p : Submodule S R
      ⊢ Membership.mem (HPow.hPow p 0) 1
    -/
    rw [← one_le, pow_zero]
    /-
      🎉 no goals
    -/
  mul_mem i j p q hp hq := by
    /-
      ι : Type u_1
      σ : Type u_2
      S : Type u_3
      R : Type u_4
      inst✝³ : DecidableEq ι
      inst✝² : CommSemiring S
      inst✝¹ : Semiring R
      inst✝ : Algebra S R
      p✝ : Submodule S R
      i j : Nat
      p q : R
      hp : Membership.mem (HPow.hPow p✝ i) p
      hq : Membership.mem (HPow.hPow p✝ j) q
      ⊢ Membership.mem (HPow.hPow p✝ (HAdd.hAdd i j)) (HMul.hMul p q)
    -/
    rw [pow_add]
    /-
      ι : Type u_1
      σ : Type u_2
      S : Type u_3
      R : Type u_4
      inst✝³ : DecidableEq ι
      inst✝² : CommSemiring S
      inst✝¹ : Semiring R
      inst✝ : Algebra S R
      p✝ : Submodule S R
      i j : Nat
      p q : R
      hp : Membership.mem (HPow.hPow p✝ i) p
      hq : Membership.mem (HPow.hPow p✝ j) q
      ⊢ Membership.mem (HMul.hMul (HPow.hPow p✝ i) (HPow.hPow p✝ j)) (HMul.hMul p q)
    -/
    exact Submodule.mul_mem_mul hp hq
    /-
      🎉 no goals
    -/


/-- The canonical algebra isomorphism between `⨁ i, A i` and `R`. -/
def DirectSum.coeAlgHom [AddMonoid ι] [CommSemiring S] [Semiring R] [Algebra S R]
    (A : ι → Submodule S R) [SetLike.GradedMonoid A] : (⨁ i, A i) →ₐ[S] R :=
  DirectSum.toAlgebra S _ (fun i => (A i).subtype) rfl (fun _ _ => rfl)


/-- The supremum of submodules that form a graded monoid is a subalgebra, and equal to the range of
`DirectSum.coeAlgHom`. -/
theorem Submodule.iSup_eq_toSubmodule_range [AddMonoid ι] [CommSemiring S] [Semiring R]
    [Algebra S R] (A : ι → Submodule S R) [SetLike.GradedMonoid A] :
    ⨆ i, A i = Subalgebra.toSubmodule (DirectSum.coeAlgHom A).range :=
  (Submodule.iSup_eq_range_dfinsupp_lsum A).trans <| SetLike.coe_injective rfl


@[simp]
theorem DirectSum.coeAlgHom_of [AddMonoid ι] [CommSemiring S] [Semiring R] [Algebra S R]
    (A : ι → Submodule S R) [SetLike.GradedMonoid A] (i : ι) (x : A i) :
    DirectSum.coeAlgHom A (DirectSum.of (fun i => A i) i x) = x :=
                                /-
                                  ι : Type u_1
                                  S : Type u_3
                                  R : Type u_4
                                  inst✝⁵ : DecidableEq ι
                                  inst✝⁴ : AddMonoid ι
                                  inst✝³ : CommSemiring S
                                  inst✝² : Semiring R
                                  inst✝¹ : Algebra S R
                                  A : ι → Submodule S R
                                  inst✝ : SetLike.GradedMonoid A
                                  i : ι
                                  x : Subtype fun x => Membership.mem (A i) x
                                  ⊢ Eq (((fun i => (A i).subtype) 0).toAddMonoidHom GradedMonoid.GOne.one) 1
                                -/
                                /-
                                  🎉 no goals
                                -/
  DirectSum.toSemiring_of _ (by rfl) (fun _ _ => (by rfl)) _ _
                                                     /-
                                                       🎉 no goals
                                                     -/


/-- The subsemiring `A 0` of `R`. -/
def subsemiring : Subsemiring R where
  carrier := A 0
  __ := submonoid A
  add_mem' := add_mem
  zero_mem' := zero_mem (A 0)

-- TODO: it might be expensive to unify `A` in this instance in practice

/-- The semiring `A 0` inherited from `R` in the presence of `SetLike.GradedMonoid A`. -/
instance instSemiring : Semiring (A 0) := (subsemiring A).toSemiring


@[simp, norm_cast] theorem coe_natCast (n : ℕ) : (n : A 0) = (n : R) := rfl


@[simp, norm_cast] theorem coe_ofNat (n : ℕ) [n.AtLeastTwo] :
    (no_index (OfNat.ofNat n) : A 0) = (OfNat.ofNat n : R) := rfl


/--The commutative semiring `A 0` inherited from `R` in the presence of `SetLike.GradedMonoid A`.-/
instance instCommSemiring : CommSemiring (A 0) := (subsemiring A).toCommSemiring


/-- The subring `A 0` of `R`. -/
def subring : Subring R where
  carrier := A 0
  __ := subsemiring A
  neg_mem' := neg_mem

-- TODO: it might be expensive to unify `A` in this instances in practice

/-- The ring `A 0` inherited from `R` in the presence of `SetLike.GradedMonoid A`. -/
instance instRing : Ring (A 0) := (subring A).toRing


theorem coe_intCast (z : ℤ) : (z : A 0) = (z : R) := rfl


/-- The commutative ring `A 0` inherited from `R` in the presence of `SetLike.GradedMonoid A`. -/
instance instCommRing : CommRing (A 0) := (subring A).toCommRing


/-- The subalgebra `A 0` of `R`. -/
def subalgebra : Subalgebra S R where
  carrier := A 0
  __ := subsemiring A
  algebraMap_mem' := algebraMap_mem_graded A

-- TODO: it might be expensive to unify `A` in this instances in practice

/-- The `S`-algebra `A 0` inherited from `R` in the presence of `SetLike.GradedMonoid A`. -/
instance instAlgebra : Algebra S (A 0) := inferInstanceAs <| Algebra S (subalgebra A)


@[simp, norm_cast] theorem coe_algebraMap (s : S) :
    ↑(algebraMap _ (A 0) s) = algebraMap _ R s := rfl


theorem SetLike.homogeneous_zero_submodule [Zero ι] [Semiring S] [AddCommMonoid R] [Module S R]
    (A : ι → Submodule S R) : SetLike.Homogeneous A (0 : R) :=
  ⟨0, Submodule.zero_mem _⟩


theorem SetLike.Homogeneous.smul [CommSemiring S] [Semiring R] [Algebra S R] {A : ι → Submodule S R}
    {s : S} {r : R} (hr : SetLike.Homogeneous A r) : SetLike.Homogeneous A (s • r) :=
  let ⟨i, hi⟩ := hr
  ⟨i, Submodule.smul_mem _ _ hi⟩


