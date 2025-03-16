local notation "ε " σ:arg => ((sign σ : ℤ) : R)


/-- `det` is an `AlternatingMap` in the rows of the matrix. -/
def detRowAlternating : (n → R) [⋀^n]→ₗ[R] R :=
  MultilinearMap.alternatization ((MultilinearMap.mkPiAlgebra R n R).compLinearMap LinearMap.proj)


/-- The determinant of a matrix given by the Leibniz formula. -/
abbrev det (M : Matrix n n R) : R :=
  detRowAlternating M


theorem det_apply (M : Matrix n n R) : M.det = ∑ σ : Perm n, Equiv.Perm.sign σ • ∏ i, M (σ i) i :=
  MultilinearMap.alternatization_apply _ M

-- This is what the old definition was. We use it to avoid having to change the old proofs below

theorem det_apply' (M : Matrix n n R) : M.det = ∑ σ : Perm n, ε σ * ∏ i, M (σ i) i := by
  /-
    n : Type u_2
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    R : Type v
    inst✝ : CommRing R
    M : Matrix n n R
    ⊢ Eq M.det (Finset.univ.sum fun σ => HMul.hMul (↑↑(Equiv.Perm.sign σ)) (Finset …
  -/
  simp [det_apply, Units.smul_def]
  /-
    🎉 no goals
  -/


theorem det_eq_detp_sub_detp (M : Matrix n n R) : M.det = M.detp 1 - M.detp (-1) := by
  /-
    n : Type u_2
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    R : Type v
    inst✝ : CommRing R
    M : Matrix n n R
    ⊢ Eq M.det (HSub.hSub (Matrix.detp 1 M) (Matrix.detp (-1) M))
  -/
  rw [det_apply, ← Equiv.sum_comp (Equiv.inv (Perm n)), ← ofSign_disjUnion, sum_disjUnion]
  /-
    n : Type u_2
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    R : Type v
    inst✝ : CommRing R
    M : Matrix n n R
    ⊢ Eq (HAdd.hAdd ((Equiv.Perm.ofSign 1).sum fun x => HSMul.hSMul (Equiv.Perm.si …
  -/
  simp_rw [inv_apply, sign_inv, sub_eq_add_neg, detp, ← sum_neg_distrib]
  /-
    n : Type u_2
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    R : Type v
    inst✝ : CommRing R
    M : Matrix n n R
    ⊢ Eq (HAdd.hAdd ((Equiv.Perm.ofSign 1).sum fun x => HSMul.hSMul (Equiv.Perm.si …
  -/
  refine congr_arg₂ (· + ·) (sum_congr rfl fun σ hσ ↦ ?_) (sum_congr rfl fun σ hσ ↦ ?_) <;>
    /-
      case refine_1
      n : Type u_2
      inst✝² : DecidableEq n
      inst✝¹ : Fintype n
      R : Type v
      inst✝ : CommRing R
      M : Matrix n n R
      σ : Equiv.Perm n
      hσ : Membership.mem (Equiv.Perm.ofSign 1) σ
      ⊢ Eq (HSMul.hSMul (Equiv.Perm.sign σ) (Finset.univ.prod fun x => M ((Inv.inv σ …
    -/
                                                   /-
                                                     🎉 no goals
                                                   -/
    rw [mem_ofSign.mp hσ, ← Equiv.prod_comp σ] <;> simp
                                                   /-
                                                     🎉 no goals
                                                   -/


@[simp]
theorem det_diagonal {d : n → R} : det (diagonal d) = ∏ i, d i := by
  /-
    n : Type u_2
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    R : Type v
    inst✝ : CommRing R
    d : n → R
    ⊢ Eq (Matrix.diagonal d).det (Finset.univ.prod fun i => d i)
  -/
  rw [det_apply']
  /-
    n : Type u_2
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    R : Type v
    inst✝ : CommRing R
    d : n → R
    ⊢ Eq (Finset.univ.sum fun σ => HMul.hMul (↑↑(Equiv.Perm.sign σ)) (Finset.univ. …
  -/
  refine (Finset.sum_eq_single 1 ?_ ?_).trans ?_
    /-
      case refine_1
      n : Type u_2
      inst✝² : DecidableEq n
      inst✝¹ : Fintype n
      R : Type v
      inst✝ : CommRing R
      d : n → R
      ⊢ ∀ (b : Equiv.Perm n), Membership.mem Finset.univ b → Ne b 1 → Eq (HMul.hMul  …
    -/
  · rintro σ - h2
    /-
      case refine_1
      n : Type u_2
      inst✝² : DecidableEq n
      inst✝¹ : Fintype n
      R : Type v
      inst✝ : CommRing R
      d : n → R
      σ : Equiv.Perm n
      h2 : Ne σ 1
      ⊢ Eq (HMul.hMul (↑↑(Equiv.Perm.sign σ)) (Finset.univ.prod fun i => Matrix.diag …
    -/
    cases' not_forall.1 (mt Equiv.ext h2) with x h3
    /-
      case refine_1.intro
      n : Type u_2
      inst✝² : DecidableEq n
      inst✝¹ : Fintype n
      R : Type v
      inst✝ : CommRing R
      d : n → R
      σ : Equiv.Perm n
      h2 : Ne σ 1
      x : n
      h3 : Not (Eq (σ x) (1 x))
      ⊢ Eq (HMul.hMul (↑↑(Equiv.Perm.sign σ)) (Finset.univ.prod fun i => Matrix.diag …
    -/
    convert mul_zero (ε σ)
    /-
      case h.e'_2.h.e'_6
      n : Type u_2
      inst✝² : DecidableEq n
      inst✝¹ : Fintype n
      R : Type v
      inst✝ : CommRing R
      d : n → R
      σ : Equiv.Perm n
      h2 : Ne σ 1
      x : n
      h3 : Not (Eq (σ x) (1 x))
      ⊢ Eq (Finset.univ.prod fun i => Matrix.diagonal d (σ i) i) 0
    -/
    apply Finset.prod_eq_zero (mem_univ x)
    /-
      case h.e'_2.h.e'_6
      n : Type u_2
      inst✝² : DecidableEq n
      inst✝¹ : Fintype n
      R : Type v
      inst✝ : CommRing R
      d : n → R
      σ : Equiv.Perm n
      h2 : Ne σ 1
      x : n
      h3 : Not (Eq (σ x) (1 x))
      ⊢ Eq (Matrix.diagonal d (σ x) x) 0
    -/
    exact if_neg h3
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      n : Type u_2
      inst✝² : DecidableEq n
      inst✝¹ : Fintype n
      R : Type v
      inst✝ : CommRing R
      d : n → R
      ⊢ Not (Membership.mem Finset.univ 1) → Eq (HMul.hMul (↑↑(Equiv.Perm.sign 1)) ( …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      n : Type u_2
      inst✝² : DecidableEq n
      inst✝¹ : Fintype n
      R : Type v
      inst✝ : CommRing R
      d : n → R
      ⊢ Eq (HMul.hMul (↑↑(Equiv.Perm.sign 1)) (Finset.univ.prod fun i => Matrix.diag …
    -/
  · simp
    /-
      🎉 no goals
    -/


theorem det_zero (_ : Nonempty n) : det (0 : Matrix n n R) = 0 :=
  (detRowAlternating : (n → R) [⋀^n]→ₗ[R] R).map_zero


@[simp]
                                                   /-
                                                     n : Type u_2
                                                     inst✝² : DecidableEq n
                                                     inst✝¹ : Fintype n
                                                     R : Type v
                                                     inst✝ : CommRing R
                                                     ⊢ Eq (Matrix.det 1) 1
                                                   -/
theorem det_one : det (1 : Matrix n n R) = 1 := by rw [← diagonal_one]; simp [-diagonal_one]
                                                                        /-
                                                                          🎉 no goals
                                                                        -/


                                                                     /-
                                                                       n : Type u_2
                                                                       inst✝³ : DecidableEq n
                                                                       inst✝² : Fintype n
                                                                       R : Type v
                                                                       inst✝¹ : CommRing R
                                                                       inst✝ : IsEmpty n
                                                                       A : Matrix n n R
                                                                       ⊢ Eq A.det 1
                                                                     -/
theorem det_isEmpty [IsEmpty n] {A : Matrix n n R} : det A = 1 := by simp [det_apply]
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


@[simp]
theorem coe_det_isEmpty [IsEmpty n] : (det : Matrix n n R → R) = Function.const _ 1 := by
  /-
    n : Type u_2
    inst✝³ : DecidableEq n
    inst✝² : Fintype n
    R : Type v
    inst✝¹ : CommRing R
    inst✝ : IsEmpty n
    ⊢ Eq Matrix.det (Function.const (Matrix n n R) 1)
  -/
  ext
  /-
    case h
    n : Type u_2
    inst✝³ : DecidableEq n
    inst✝² : Fintype n
    R : Type v
    inst✝¹ : CommRing R
    inst✝ : IsEmpty n
    x✝ : Matrix n n R
    ⊢ Eq x✝.det (Function.const (Matrix n n R) 1 x✝)
  -/
  exact det_isEmpty
  /-
    🎉 no goals
  -/


theorem det_eq_one_of_card_eq_zero {A : Matrix n n R} (h : Fintype.card n = 0) : det A = 1 :=
  haveI : IsEmpty n := Fintype.card_eq_zero_iff.mp h
  det_isEmpty


/-- If `n` has only one element, the determinant of an `n` by `n` matrix is just that element.
Although `Unique` implies `DecidableEq` and `Fintype`, the instances might
not be syntactically equal. Thus, we need to fill in the args explicitly. -/
@[simp]
theorem det_unique {n : Type*} [Unique n] [DecidableEq n] [Fintype n] (A : Matrix n n R) :
                                    /-
                                      R : Type v
                                      inst✝³ : CommRing R
                                      n : Type u_3
                                      inst✝² : Unique n
                                      inst✝¹ : DecidableEq n
                                      inst✝ : Fintype n
                                      A : Matrix n n R
                                      ⊢ Eq A.det (A Inhabited.default Inhabited.default)
                                    -/
    det A = A default default := by simp [det_apply, univ_unique]
                                    /-
                                      🎉 no goals
                                    -/


theorem det_eq_elem_of_subsingleton [Subsingleton n] (A : Matrix n n R) (k : n) :
    det A = A k k := by
  /-
    n : Type u_2
    inst✝³ : DecidableEq n
    inst✝² : Fintype n
    R : Type v
    inst✝¹ : CommRing R
    inst✝ : Subsingleton n
    A : Matrix n n R
    k : n
    ⊢ Eq A.det (A k k)
  -/
  have := uniqueOfSubsingleton k
  /-
    n : Type u_2
    inst✝³ : DecidableEq n
    inst✝² : Fintype n
    R : Type v
    inst✝¹ : CommRing R
    inst✝ : Subsingleton n
    A : Matrix n n R
    k : n
    this : Unique n
    ⊢ Eq A.det (A k k)
  -/
  convert det_unique A
  /-
    🎉 no goals
  -/


theorem det_eq_elem_of_card_eq_one {A : Matrix n n R} (h : Fintype.card n = 1) (k : n) :
    det A = A k k :=
  haveI : Subsingleton n := Fintype.card_le_one_iff_subsingleton.mp h.le
  det_eq_elem_of_subsingleton _ _


theorem det_mul_aux {M N : Matrix n n R} {p : n → n} (H : ¬Bijective p) :
    (∑ σ : Perm n, ε σ * ∏ x, M (σ x) (p x) * N (p x) x) = 0 := by
  obtain ⟨i, j, hpij, hij⟩ : ∃ i j, p i = p j ∧ i ≠ j := by
    rw [← Finite.injective_iff_bijective, Injective] at H
    push_neg at H
    exact H
  exact
    sum_involution (fun σ _ => σ * Equiv.swap i j)
      (fun σ _ => by
        have : (∏ x, M (σ x) (p x)) = ∏ x, M ((σ * Equiv.swap i j) x) (p x) :=
          Fintype.prod_equiv (swap i j) _ _ (by simp [apply_swap_eq_self hpij])
        simp [this, sign_swap hij, -sign_swap', prod_mul_distrib])
      (fun σ _ _ => (not_congr mul_swap_eq_iff).mpr hij) (fun _ _ => mem_univ _) fun σ _ =>
      mul_swap_involutive i j σ


@[simp]
theorem det_mul (M N : Matrix n n R) : det (M * N) = det M * det N :=
  calc
    det (M * N) = ∑ p : n → n, ∑ σ : Perm n, ε σ * ∏ i, M (σ i) (p i) * N (p i) i := by
      /-
        n : Type u_2
        inst✝² : DecidableEq n
        inst✝¹ : Fintype n
        R : Type v
        inst✝ : CommRing R
        M N : Matrix n n R
        ⊢ Eq (HMul.hMul M N).det (Finset.univ.sum fun p => Finset.univ.sum fun σ => HM …
      -/
      simp only [det_apply', mul_apply, prod_univ_sum, mul_sum, Fintype.piFinset_univ]
      /-
        n : Type u_2
        inst✝² : DecidableEq n
        inst✝¹ : Fintype n
        R : Type v
        inst✝ : CommRing R
        M N : Matrix n n R
        ⊢ Eq (Finset.univ.sum fun x => Finset.univ.sum fun i => HMul.hMul (↑↑(Equiv.Pe …
      -/
      rw [Finset.sum_comm]
      /-
        🎉 no goals
      -/
    _ = ∑ p : n → n with Bijective p, ∑ σ : Perm n, ε σ * ∏ i, M (σ i) (p i) * N (p i) i := by
      /-
        n : Type u_2
        inst✝² : DecidableEq n
        inst✝¹ : Fintype n
        R : Type v
        inst✝ : CommRing R
        M N : Matrix n n R
        ⊢ Eq (Finset.univ.sum fun p => Finset.univ.sum fun σ => HMul.hMul (↑↑(Equiv.Pe …
      -/
      refine (sum_subset (filter_subset _ _) fun f _ hbij ↦ det_mul_aux ?_).symm
      /-
        n : Type u_2
        inst✝² : DecidableEq n
        inst✝¹ : Fintype n
        R : Type v
        inst✝ : CommRing R
        M N : Matrix n n R
        f : n → n
        x✝ : Membership.mem Finset.univ f
        hbij : Not (Membership.mem (Finset.filter (fun p => Function.Bijective p) Fins …
        ⊢ Not (Function.Bijective f)
      -/
      simpa only [true_and, mem_filter, mem_univ] using hbij
      /-
        🎉 no goals
      -/
    _ = ∑ τ : Perm n, ∑ σ : Perm n, ε σ * ∏ i, M (σ i) (τ i) * N (τ i) i :=
      sum_bij (fun p h ↦ Equiv.ofBijective p (mem_filter.1 h).2) (fun _ _ ↦ mem_univ _)
                            /-
                              n : Type u_2
                              inst✝² : DecidableEq n
                              inst✝¹ : Fintype n
                              R : Type v
                              inst✝ : CommRing R
                              M N : Matrix n n R
                              x✝³ : n → n
                              x✝² : Membership.mem (Finset.filter (fun p => Function.Bijective p) Finset.uni …
                              x✝¹ : n → n
                              x✝ : Membership.mem (Finset.filter (fun p => Function.Bijective p) Finset.univ …
                              h : Eq ((fun p h => Equiv.ofBijective p ⋯) x✝³ x✝²) ((fun p h => Equiv.ofBijec …
                              ⊢ Eq x✝³ x✝¹
                            -/
        (fun _ _ _ _ h ↦ by injection h)
                            /-
                              🎉 no goals
                            -/
        (fun b _ ↦ ⟨b, mem_filter.2 ⟨mem_univ _, b.bijective⟩, coe_fn_injective rfl⟩) fun _ _ ↦ rfl
    _ = ∑ σ : Perm n, ∑ τ : Perm n, (∏ i, N (σ i) i) * ε τ * ∏ j, M (τ j) (σ j) := by
      /-
        n : Type u_2
        inst✝² : DecidableEq n
        inst✝¹ : Fintype n
        R : Type v
        inst✝ : CommRing R
        M N : Matrix n n R
        ⊢ Eq (Finset.univ.sum fun τ => Finset.univ.sum fun σ => HMul.hMul (↑↑(Equiv.Pe …
      -/
      simp only [mul_comm, mul_left_comm, prod_mul_distrib, mul_assoc]
      /-
        🎉 no goals
      -/
    _ = ∑ σ : Perm n, ∑ τ : Perm n, (∏ i, N (σ i) i) * (ε σ * ε τ) * ∏ i, M (τ i) i :=
      (sum_congr rfl fun σ _ =>
        Fintype.sum_equiv (Equiv.mulRight σ⁻¹) _ _ fun τ => by
          have : (∏ j, M (τ j) (σ j)) = ∏ j, M ((τ * σ⁻¹) j) j := by
            rw [← (σ⁻¹ : _ ≃ _).prod_comp]
            simp only [Equiv.Perm.coe_mul, apply_inv_self, Function.comp_apply]
          have h : ε σ * ε (τ * σ⁻¹) = ε τ :=
            calc
              ε σ * ε (τ * σ⁻¹) = ε (τ * σ⁻¹ * σ) := by
                rw [mul_comm, sign_mul (τ * σ⁻¹)]
                simp only [Int.cast_mul, Units.val_mul]
              _ = ε τ := by simp only [inv_mul_cancel_right]

          /-
            n : Type u_2
            inst✝² : DecidableEq n
            inst✝¹ : Fintype n
            R : Type v
            inst✝ : CommRing R
            M N : Matrix n n R
            σ : Equiv.Perm n
            x✝ : Membership.mem Finset.univ σ
            τ : Equiv.Perm n
            this : Eq (Finset.univ.prod fun j => M (τ j) (σ j)) (Finset.univ.prod fun j => …
            h : Eq (HMul.hMul ↑↑(Equiv.Perm.sign σ) ↑↑(Equiv.Perm.sign (HMul.hMul τ (Inv.i …
            ⊢ Eq (HMul.hMul (HMul.hMul (Finset.univ.prod fun i => N (σ i) i) ↑↑(Equiv.Perm …
          -/
          simp_rw [Equiv.coe_mulRight, h]
          /-
            n : Type u_2
            inst✝² : DecidableEq n
            inst✝¹ : Fintype n
            R : Type v
            inst✝ : CommRing R
            M N : Matrix n n R
            σ : Equiv.Perm n
            x✝ : Membership.mem Finset.univ σ
            τ : Equiv.Perm n
            this : Eq (Finset.univ.prod fun j => M (τ j) (σ j)) (Finset.univ.prod fun j => …
            h : Eq (HMul.hMul ↑↑(Equiv.Perm.sign σ) ↑↑(Equiv.Perm.sign (HMul.hMul τ (Inv.i …
            ⊢ Eq (HMul.hMul (HMul.hMul (Finset.univ.prod fun i => N (σ i) i) ↑↑(Equiv.Perm …
          -/
          simp only [this])
          /-
            🎉 no goals
          -/
    _ = det M * det N := by
      /-
        n : Type u_2
        inst✝² : DecidableEq n
        inst✝¹ : Fintype n
        R : Type v
        inst✝ : CommRing R
        M N : Matrix n n R
        ⊢ Eq (Finset.univ.sum fun σ => Finset.univ.sum fun τ => HMul.hMul (HMul.hMul ( …
      -/
      simp only [det_apply', Finset.mul_sum, mul_comm, mul_left_comm, mul_assoc]
      /-
        🎉 no goals
      -/


/-- The determinant of a matrix, as a monoid homomorphism. -/
def detMonoidHom : Matrix n n R →* R where
  toFun := det
  map_one' := det_one
  map_mul' := det_mul


@[simp]
theorem coe_detMonoidHom : (detMonoidHom : Matrix n n R → R) = det :=
  rfl


/-- On square matrices, `mul_comm` applies under `det`. -/
theorem det_mul_comm (M N : Matrix m m R) : det (M * N) = det (N * M) := by
  /-
    m : Type u_1
    inst✝² : DecidableEq m
    inst✝¹ : Fintype m
    R : Type v
    inst✝ : CommRing R
    M N : Matrix m m R
    ⊢ Eq (HMul.hMul M N).det (HMul.hMul N M).det
  -/
  rw [det_mul, det_mul, mul_comm]
  /-
    🎉 no goals
  -/


/-- On square matrices, `mul_left_comm` applies under `det`. -/
theorem det_mul_left_comm (M N P : Matrix m m R) : det (M * (N * P)) = det (N * (M * P)) := by
  /-
    m : Type u_1
    inst✝² : DecidableEq m
    inst✝¹ : Fintype m
    R : Type v
    inst✝ : CommRing R
    M N P : Matrix m m R
    ⊢ Eq (HMul.hMul M (HMul.hMul N P)).det (HMul.hMul N (HMul.hMul M P)).det
  -/
  rw [← Matrix.mul_assoc, ← Matrix.mul_assoc, det_mul, det_mul_comm M N, ← det_mul]
  /-
    🎉 no goals
  -/


/-- On square matrices, `mul_right_comm` applies under `det`. -/
theorem det_mul_right_comm (M N P : Matrix m m R) : det (M * N * P) = det (M * P * N) := by
  /-
    m : Type u_1
    inst✝² : DecidableEq m
    inst✝¹ : Fintype m
    R : Type v
    inst✝ : CommRing R
    M N P : Matrix m m R
    ⊢ Eq (HMul.hMul (HMul.hMul M N) P).det (HMul.hMul (HMul.hMul M P) N).det
  -/
  rw [Matrix.mul_assoc, Matrix.mul_assoc, det_mul, det_mul_comm N P, ← det_mul]
  /-
    🎉 no goals
  -/

-- TODO(https://github.com/leanprover-community/mathlib4/issues/6607): fix elaboration so `val` isn't needed

theorem det_units_conj (M : (Matrix m m R)ˣ) (N : Matrix m m R) :
    det (M.val * N * M⁻¹.val) = det N := by
  /-
    m : Type u_1
    inst✝² : DecidableEq m
    inst✝¹ : Fintype m
    R : Type v
    inst✝ : CommRing R
    M : Units (Matrix m m R)
    N : Matrix m m R
    ⊢ Eq (HMul.hMul (HMul.hMul (↑M) N) ↑(Inv.inv M)).det N.det
  -/
  rw [det_mul_right_comm, Units.mul_inv, one_mul]
  /-
    🎉 no goals
  -/

-- TODO(https://github.com/leanprover-community/mathlib4/issues/6607): fix elaboration so `val` isn't needed

theorem det_units_conj' (M : (Matrix m m R)ˣ) (N : Matrix m m R) :
    det (M⁻¹.val * N * ↑M.val) = det N :=
  det_units_conj M⁻¹ N


/-- Transposing a matrix preserves the determinant. -/
@[simp]
theorem det_transpose (M : Matrix n n R) : Mᵀ.det = M.det := by
  /-
    n : Type u_2
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    R : Type v
    inst✝ : CommRing R
    M : Matrix n n R
    ⊢ Eq M.transpose.det M.det
  -/
  rw [det_apply', det_apply']
  /-
    n : Type u_2
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    R : Type v
    inst✝ : CommRing R
    M : Matrix n n R
    ⊢ Eq (Finset.univ.sum fun σ => HMul.hMul (↑↑(Equiv.Perm.sign σ)) (Finset.univ. …
  -/
  refine Fintype.sum_bijective _ inv_involutive.bijective _ _ ?_
  /-
    n : Type u_2
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    R : Type v
    inst✝ : CommRing R
    M : Matrix n n R
    ⊢ ∀ (x : Equiv.Perm n), Eq (HMul.hMul (↑↑(Equiv.Perm.sign x)) (Finset.univ.pro …
  -/
  intro σ
  /-
    n : Type u_2
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    R : Type v
    inst✝ : CommRing R
    M : Matrix n n R
    σ : Equiv.Perm n
    ⊢ Eq (HMul.hMul (↑↑(Equiv.Perm.sign σ)) (Finset.univ.prod fun i => M.transpose …
  -/
  rw [sign_inv]
  /-
    n : Type u_2
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    R : Type v
    inst✝ : CommRing R
    M : Matrix n n R
    σ : Equiv.Perm n
    ⊢ Eq (HMul.hMul (↑↑(Equiv.Perm.sign σ)) (Finset.univ.prod fun i => M.transpose …
  -/
  congr 1
  /-
    case e_a
    n : Type u_2
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    R : Type v
    inst✝ : CommRing R
    M : Matrix n n R
    σ : Equiv.Perm n
    ⊢ Eq (Finset.univ.prod fun i => M.transpose (σ i) i) (Finset.univ.prod fun i = …
  -/
  apply Fintype.prod_equiv σ
  /-
    case e_a.h
    n : Type u_2
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    R : Type v
    inst✝ : CommRing R
    M : Matrix n n R
    σ : Equiv.Perm n
    ⊢ ∀ (x : n), Eq (M.transpose (σ x) x) (M ((Inv.inv σ) (σ x)) (σ x))
  -/
  simp
  /-
    🎉 no goals
  -/


/-- Permuting the columns changes the sign of the determinant. -/
theorem det_permute (σ : Perm n) (M : Matrix n n R) :
    (M.submatrix σ id).det = Perm.sign σ * M.det :=
                                                                      /-
                                                                        n : Type u_2
                                                                        inst✝² : DecidableEq n
                                                                        inst✝¹ : Fintype n
                                                                        R : Type v
                                                                        inst✝ : CommRing R
                                                                        σ : Equiv.Perm n
                                                                        M : Matrix n n R
                                                                        ⊢ Eq (HSMul.hSMul (Equiv.Perm.sign σ) (Matrix.detRowAlternating M)) (HMul.hMul …
                                                                      -/
  ((detRowAlternating : (n → R) [⋀^n]→ₗ[R] R).map_perm M σ).trans (by simp [Units.smul_def])
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


/-- Permuting the rows changes the sign of the determinant. -/
theorem det_permute' (σ : Perm n) (M : Matrix n n R) :
    (M.submatrix id σ).det = Perm.sign σ * M.det := by
  /-
    n : Type u_2
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    R : Type v
    inst✝ : CommRing R
    σ : Equiv.Perm n
    M : Matrix n n R
    ⊢ Eq (M.submatrix id ⇑σ).det (HMul.hMul (↑↑(Equiv.Perm.sign σ)) M.det)
  -/
  rw [← det_transpose, transpose_submatrix, det_permute, det_transpose]
  /-
    🎉 no goals
  -/


/-- Permuting rows and columns with the same equivalence does not change the determinant. -/
@[simp]
theorem det_submatrix_equiv_self (e : n ≃ m) (A : Matrix m m R) :
    det (A.submatrix e e) = det A := by
  /-
    m : Type u_1
    n : Type u_2
    inst✝⁴ : DecidableEq n
    inst✝³ : Fintype n
    inst✝² : DecidableEq m
    inst✝¹ : Fintype m
    R : Type v
    inst✝ : CommRing R
    e : Equiv n m
    A : Matrix m m R
    ⊢ Eq (A.submatrix ⇑e ⇑e).det A.det
  -/
  rw [det_apply', det_apply']
  /-
    m : Type u_1
    n : Type u_2
    inst✝⁴ : DecidableEq n
    inst✝³ : Fintype n
    inst✝² : DecidableEq m
    inst✝¹ : Fintype m
    R : Type v
    inst✝ : CommRing R
    e : Equiv n m
    A : Matrix m m R
    ⊢ Eq (Finset.univ.sum fun σ => HMul.hMul (↑↑(Equiv.Perm.sign σ)) (Finset.univ. …
  -/
  apply Fintype.sum_equiv (Equiv.permCongr e)
  /-
    case h
    m : Type u_1
    n : Type u_2
    inst✝⁴ : DecidableEq n
    inst✝³ : Fintype n
    inst✝² : DecidableEq m
    inst✝¹ : Fintype m
    R : Type v
    inst✝ : CommRing R
    e : Equiv n m
    A : Matrix m m R
    ⊢ ∀ (x : Equiv.Perm n), Eq (HMul.hMul (↑↑(Equiv.Perm.sign x)) (Finset.univ.pro …
  -/
  intro σ
  /-
    case h
    m : Type u_1
    n : Type u_2
    inst✝⁴ : DecidableEq n
    inst✝³ : Fintype n
    inst✝² : DecidableEq m
    inst✝¹ : Fintype m
    R : Type v
    inst✝ : CommRing R
    e : Equiv n m
    A : Matrix m m R
    σ : Equiv.Perm n
    ⊢ Eq (HMul.hMul (↑↑(Equiv.Perm.sign σ)) (Finset.univ.prod fun i => A.submatrix …
  -/
  rw [Equiv.Perm.sign_permCongr e σ]
  /-
    case h
    m : Type u_1
    n : Type u_2
    inst✝⁴ : DecidableEq n
    inst✝³ : Fintype n
    inst✝² : DecidableEq m
    inst✝¹ : Fintype m
    R : Type v
    inst✝ : CommRing R
    e : Equiv n m
    A : Matrix m m R
    σ : Equiv.Perm n
    ⊢ Eq (HMul.hMul (↑↑(Equiv.Perm.sign σ)) (Finset.univ.prod fun i => A.submatrix …
  -/
  congr 1
  /-
    case h.e_a
    m : Type u_1
    n : Type u_2
    inst✝⁴ : DecidableEq n
    inst✝³ : Fintype n
    inst✝² : DecidableEq m
    inst✝¹ : Fintype m
    R : Type v
    inst✝ : CommRing R
    e : Equiv n m
    A : Matrix m m R
    σ : Equiv.Perm n
    ⊢ Eq (Finset.univ.prod fun i => A.submatrix (⇑e) (⇑e) (σ i) i) (Finset.univ.pr …
  -/
  apply Fintype.prod_equiv e
  /-
    case h.e_a.h
    m : Type u_1
    n : Type u_2
    inst✝⁴ : DecidableEq n
    inst✝³ : Fintype n
    inst✝² : DecidableEq m
    inst✝¹ : Fintype m
    R : Type v
    inst✝ : CommRing R
    e : Equiv n m
    A : Matrix m m R
    σ : Equiv.Perm n
    ⊢ ∀ (x : n), Eq (A.submatrix (⇑e) (⇑e) (σ x) x) (A ((e.permCongr σ) (e x)) (e  …
  -/
  intro i
  /-
    case h.e_a.h
    m : Type u_1
    n : Type u_2
    inst✝⁴ : DecidableEq n
    inst✝³ : Fintype n
    inst✝² : DecidableEq m
    inst✝¹ : Fintype m
    R : Type v
    inst✝ : CommRing R
    e : Equiv n m
    A : Matrix m m R
    σ : Equiv.Perm n
    i : n
    ⊢ Eq (A.submatrix (⇑e) (⇑e) (σ i) i) (A ((e.permCongr σ) (e i)) (e i))
  -/
  rw [Equiv.permCongr_apply, Equiv.symm_apply_apply, submatrix_apply]
  /-
    🎉 no goals
  -/


/-- Permuting rows and columns with two equivalences does not change the absolute value of the
determinant. -/
@[simp]
theorem abs_det_submatrix_equiv_equiv {R : Type*} [LinearOrderedCommRing R]
    (e₁ e₂ : n ≃ m) (A : Matrix m m R) :
    |(A.submatrix e₁ e₂).det| = |A.det| := by
  /-
    m : Type u_1
    n : Type u_2
    inst✝⁴ : DecidableEq n
    inst✝³ : Fintype n
    inst✝² : DecidableEq m
    inst✝¹ : Fintype m
    R : Type u_3
    inst✝ : LinearOrderedCommRing R
    e₁ e₂ : Equiv n m
    A : Matrix m m R
    ⊢ Eq (abs (A.submatrix ⇑e₁ ⇑e₂).det) (abs A.det)
  -/
  have hee : e₂ = e₁.trans (e₁.symm.trans e₂) := by ext; simp
  /-
    m : Type u_1
    n : Type u_2
    inst✝⁴ : DecidableEq n
    inst✝³ : Fintype n
    inst✝² : DecidableEq m
    inst✝¹ : Fintype m
    R : Type u_3
    inst✝ : LinearOrderedCommRing R
    e₁ e₂ : Equiv n m
    A : Matrix m m R
    hee : Eq e₂ (e₁.trans (e₁.symm.trans e₂))
    ⊢ Eq (abs (A.submatrix ⇑e₁ ⇑e₂).det) (abs A.det)
  -/
  rw [hee]
  /-
    m : Type u_1
    n : Type u_2
    inst✝⁴ : DecidableEq n
    inst✝³ : Fintype n
    inst✝² : DecidableEq m
    inst✝¹ : Fintype m
    R : Type u_3
    inst✝ : LinearOrderedCommRing R
    e₁ e₂ : Equiv n m
    A : Matrix m m R
    hee : Eq e₂ (e₁.trans (e₁.symm.trans e₂))
    ⊢ Eq (abs (A.submatrix ⇑e₁ ⇑(e₁.trans (e₁.symm.trans e₂))).det) (abs A.det)
  -/
  show |((A.submatrix id (e₁.symm.trans e₂)).submatrix e₁ e₁).det| = |A.det|
  /-
    m : Type u_1
    n : Type u_2
    inst✝⁴ : DecidableEq n
    inst✝³ : Fintype n
    inst✝² : DecidableEq m
    inst✝¹ : Fintype m
    R : Type u_3
    inst✝ : LinearOrderedCommRing R
    e₁ e₂ : Equiv n m
    A : Matrix m m R
    hee : Eq e₂ (e₁.trans (e₁.symm.trans e₂))
    ⊢ Eq (abs ((A.submatrix id ⇑(e₁.symm.trans e₂)).submatrix ⇑e₁ ⇑e₁).det) (abs A …
  -/
  rw [Matrix.det_submatrix_equiv_self, Matrix.det_permute', abs_mul, abs_unit_intCast, one_mul]
  /-
    🎉 no goals
  -/


/-- Reindexing both indices along the same equivalence preserves the determinant.

For the `simp` version of this lemma, see `det_submatrix_equiv_self`; this one is unsuitable because
`Matrix.reindex_apply` unfolds `reindex` first.
-/
theorem det_reindex_self (e : m ≃ n) (A : Matrix m m R) : det (reindex e e A) = det A :=
  det_submatrix_equiv_self e.symm A


theorem det_smul (A : Matrix n n R) (c : R) : det (c • A) = c ^ Fintype.card n * det A :=
  calc
                                                        /-
                                                          n : Type u_2
                                                          inst✝² : DecidableEq n
                                                          inst✝¹ : Fintype n
                                                          R : Type v
                                                          inst✝ : CommRing R
                                                          A : Matrix n n R
                                                          c : R
                                                          ⊢ Eq (HSMul.hSMul c A).det (HMul.hMul (Matrix.diagonal fun x => c) A).det
                                                        -/
    det (c • A) = det ((diagonal fun _ => c) * A) := by rw [smul_eq_diagonal_mul]
                                                        /-
                                                          🎉 no goals
                                                        -/
    _ = det (diagonal fun _ => c) * det A := det_mul _ _
                                         /-
                                           n : Type u_2
                                           inst✝² : DecidableEq n
                                           inst✝¹ : Fintype n
                                           R : Type v
                                           inst✝ : CommRing R
                                           A : Matrix n n R
                                           c : R
                                           ⊢ Eq (HMul.hMul (Matrix.diagonal fun x => c).det A.det) (HMul.hMul (HPow.hPow  …
                                         -/
    _ = c ^ Fintype.card n * det A := by simp [card_univ]
                                         /-
                                           🎉 no goals
                                         -/


@[simp]
theorem det_smul_of_tower {α} [Monoid α] [DistribMulAction α R] [IsScalarTower α R R]
    [SMulCommClass α R R] (c : α) (A : Matrix n n R) :
    det (c • A) = c ^ Fintype.card n • det A := by
  /-
    n : Type u_2
    inst✝⁶ : DecidableEq n
    inst✝⁵ : Fintype n
    R : Type v
    inst✝⁴ : CommRing R
    α : Type u_3
    inst✝³ : Monoid α
    inst✝² : DistribMulAction α R
    inst✝¹ : IsScalarTower α R R
    inst✝ : SMulCommClass α R R
    c : α
    A : Matrix n n R
    ⊢ Eq (HSMul.hSMul c A).det (HSMul.hSMul (HPow.hPow c (Fintype.card n)) A.det)
  -/
  rw [← smul_one_smul R c A, det_smul, smul_pow, one_pow, smul_mul_assoc, one_mul]
  /-
    🎉 no goals
  -/


theorem det_neg (A : Matrix n n R) : det (-A) = (-1) ^ Fintype.card n * det A := by
  /-
    n : Type u_2
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    R : Type v
    inst✝ : CommRing R
    A : Matrix n n R
    ⊢ Eq (Neg.neg A).det (HMul.hMul (HPow.hPow (-1) (Fintype.card n)) A.det)
  -/
  rw [← det_smul, neg_one_smul]
  /-
    🎉 no goals
  -/


/-- A variant of `Matrix.det_neg` with scalar multiplication by `Units ℤ` instead of multiplication
by `R`. -/
theorem det_neg_eq_smul (A : Matrix n n R) :
    det (-A) = (-1 : Units ℤ) ^ Fintype.card n • det A := by
  /-
    n : Type u_2
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    R : Type v
    inst✝ : CommRing R
    A : Matrix n n R
    ⊢ Eq (Neg.neg A).det (HSMul.hSMul (HPow.hPow (-1) (Fintype.card n)) A.det)
  -/
  rw [← det_smul_of_tower, Units.neg_smul, one_smul]
  /-
    🎉 no goals
  -/


/-- Multiplying each row by a fixed `v i` multiplies the determinant by
the product of the `v`s. -/
theorem det_mul_row (v : n → R) (A : Matrix n n R) :
    det (of fun i j => v j * A i j) = (∏ i, v i) * det A :=
  calc
    det (of fun i j => v j * A i j) = det (A * diagonal v) :=
      congr_arg det <| by
        /-
          n : Type u_2
          inst✝² : DecidableEq n
          inst✝¹ : Fintype n
          R : Type v
          inst✝ : CommRing R
          v : n → R
          A : Matrix n n R
          ⊢ Eq (Matrix.of fun i j => HMul.hMul (v j) (A i j)) (HMul.hMul A (Matrix.diago …
        -/
        ext
        /-
          case a
          n : Type u_2
          inst✝² : DecidableEq n
          inst✝¹ : Fintype n
          R : Type v
          inst✝ : CommRing R
          v : n → R
          A : Matrix n n R
          i✝ j✝ : n
          ⊢ Eq (Matrix.of (fun i j => HMul.hMul (v j) (A i j)) i✝ j✝) (HMul.hMul A (Matr …
        -/
        simp [mul_comm]
        /-
          🎉 no goals
        -/
                                 /-
                                   n : Type u_2
                                   inst✝² : DecidableEq n
                                   inst✝¹ : Fintype n
                                   R : Type v
                                   inst✝ : CommRing R
                                   v : n → R
                                   A : Matrix n n R
                                   ⊢ Eq (HMul.hMul A (Matrix.diagonal v)).det (HMul.hMul (Finset.univ.prod fun i  …
                                 -/
    _ = (∏ i, v i) * det A := by rw [det_mul, det_diagonal, mul_comm]
                                 /-
                                   🎉 no goals
                                 -/


/-- Multiplying each column by a fixed `v j` multiplies the determinant by
the product of the `v`s. -/
theorem det_mul_column (v : n → R) (A : Matrix n n R) :
    det (of fun i j => v i * A i j) = (∏ i, v i) * det A :=
  MultilinearMap.map_smul_univ _ v A


@[simp]
theorem det_pow (M : Matrix m m R) (n : ℕ) : det (M ^ n) = det M ^ n :=
  (detMonoidHom : Matrix m m R →* R).map_pow M n


theorem _root_.RingHom.map_det (f : R →+* S) (M : Matrix n n R) :
    f M.det = Matrix.det (f.mapMatrix M) := by
  /-
    n : Type u_2
    inst✝³ : DecidableEq n
    inst✝² : Fintype n
    R : Type v
    inst✝¹ : CommRing R
    S : Type w
    inst✝ : CommRing S
    f : RingHom R S
    M : Matrix n n R
    ⊢ Eq (f M.det) (f.mapMatrix M).det
  -/
  simp [Matrix.det_apply', map_sum f, map_prod f]
  /-
    🎉 no goals
  -/


theorem _root_.RingEquiv.map_det (f : R ≃+* S) (M : Matrix n n R) :
    f M.det = Matrix.det (f.mapMatrix M) :=
  f.toRingHom.map_det _


theorem _root_.AlgHom.map_det [Algebra R S] {T : Type z} [CommRing T] [Algebra R T] (f : S →ₐ[R] T)
    (M : Matrix n n S) : f M.det = Matrix.det (f.mapMatrix M) :=
  f.toRingHom.map_det _


theorem _root_.AlgEquiv.map_det [Algebra R S] {T : Type z} [CommRing T] [Algebra R T]
    (f : S ≃ₐ[R] T) (M : Matrix n n S) : f M.det = Matrix.det (f.mapMatrix M) :=
  f.toAlgHom.map_det _


@[simp]
theorem det_conjTranspose [StarRing R] (M : Matrix m m R) : det Mᴴ = star (det M) :=
  ((starRingEnd R).map_det _).symm.trans <| congr_arg star M.det_transpose


theorem det_eq_zero_of_row_eq_zero {A : Matrix n n R} (i : n) (h : ∀ j, A i j = 0) : det A = 0 :=
  (detRowAlternating : (n → R) [⋀^n]→ₗ[R] R).map_coord_zero i (funext h)


theorem det_eq_zero_of_column_eq_zero {A : Matrix n n R} (j : n) (h : ∀ i, A i j = 0) :
    det A = 0 := by
  /-
    n : Type u_2
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    R : Type v
    inst✝ : CommRing R
    A : Matrix n n R
    j : n
    h : ∀ (i : n), Eq (A i j) 0
    ⊢ Eq A.det 0
  -/
  rw [← det_transpose]
  /-
    n : Type u_2
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    R : Type v
    inst✝ : CommRing R
    A : Matrix n n R
    j : n
    h : ∀ (i : n), Eq (A i j) 0
    ⊢ Eq A.transpose.det 0
  -/
  exact det_eq_zero_of_row_eq_zero j h
  /-
    🎉 no goals
  -/


/-- If a matrix has a repeated row, the determinant will be zero. -/
theorem det_zero_of_row_eq (i_ne_j : i ≠ j) (hij : M i = M j) : M.det = 0 :=
  (detRowAlternating : (n → R) [⋀^n]→ₗ[R] R).map_eq_zero_of_eq M hij i_ne_j


/-- If a matrix has a repeated column, the determinant will be zero. -/
theorem det_zero_of_column_eq (i_ne_j : i ≠ j) (hij : ∀ k, M k i = M k j) : M.det = 0 := by
  /-
    n : Type u_2
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    R : Type v
    inst✝ : CommRing R
    M : Matrix n n R
    i j : n
    i_ne_j : Ne i j
    hij : ∀ (k : n), Eq (M k i) (M k j)
    ⊢ Eq M.det 0
  -/
  rw [← det_transpose, det_zero_of_row_eq i_ne_j]
  /-
    n : Type u_2
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    R : Type v
    inst✝ : CommRing R
    M : Matrix n n R
    i j : n
    i_ne_j : Ne i j
    hij : ∀ (k : n), Eq (M k i) (M k j)
    ⊢ Eq (M.transpose i) (M.transpose j)
  -/
  exact funext hij
  /-
    🎉 no goals
  -/


/-- If we repeat a row of a matrix, we get a matrix of determinant zero. -/
theorem det_updateRow_eq_zero (h : i ≠ j) :
                                                              /-
                                                                n : Type u_2
                                                                inst✝² : DecidableEq n
                                                                inst✝¹ : Fintype n
                                                                R : Type v
                                                                inst✝ : CommRing R
                                                                M : Matrix n n R
                                                                i j : n
                                                                h : Ne i j
                                                                ⊢ Eq (M.updateRow j (M i) i) (M.updateRow j (M i) j)
                                                              -/
    (M.updateRow j (M i)).det = 0 := det_zero_of_row_eq h (by simp [h])
                                                              /-
                                                                🎉 no goals
                                                              -/


/-- If we repeat a column of a matrix, we get a matrix of determinant zero. -/
theorem det_updateCol_eq_zero (h : i ≠ j) :
                                                                           /-
                                                                             n : Type u_2
                                                                             inst✝² : DecidableEq n
                                                                             inst✝¹ : Fintype n
                                                                             R : Type v
                                                                             inst✝ : CommRing R
                                                                             M : Matrix n n R
                                                                             i j : n
                                                                             h : Ne i j
                                                                             ⊢ ∀ (k : n), Eq (M.updateCol j (fun k => M k i) k i) (M.updateCol j (fun k =>  …
                                                                           -/
    (M.updateCol j (fun k ↦ M k i)).det = 0 := det_zero_of_column_eq h (by simp [h])
                                                                           /-
                                                                             🎉 no goals
                                                                           -/


@[deprecated (since := "2024-12-11")] alias det_updateColumn_eq_zero := det_updateCol_eq_zero


theorem det_updateRow_add (M : Matrix n n R) (j : n) (u v : n → R) :
    det (updateRow M j <| u + v) = det (updateRow M j u) + det (updateRow M j v) :=
  (detRowAlternating : (n → R) [⋀^n]→ₗ[R] R).map_update_add M j u v


theorem det_updateCol_add (M : Matrix n n R) (j : n) (u v : n → R) :
    det (updateCol M j <| u + v) = det (updateCol M j u) + det (updateCol M j v) := by
  /-
    n : Type u_2
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    R : Type v
    inst✝ : CommRing R
    M : Matrix n n R
    j : n
    u v : n → R
    ⊢ Eq (M.updateCol j (HAdd.hAdd u v)).det (HAdd.hAdd (M.updateCol j u).det (M.u …
  -/
  rw [← det_transpose, ← updateRow_transpose, det_updateRow_add]
  /-
    n : Type u_2
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    R : Type v
    inst✝ : CommRing R
    M : Matrix n n R
    j : n
    u v : n → R
    ⊢ Eq (HAdd.hAdd (M.transpose.updateRow j u).det (M.transpose.updateRow j v).de …
  -/
  simp [updateRow_transpose, det_transpose]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-12-11")] alias det_updateColumn_add := det_updateCol_add


theorem det_updateRow_smul (M : Matrix n n R) (j : n) (s : R) (u : n → R) :
    det (updateRow M j <| s • u) = s * det (updateRow M j u) :=
  (detRowAlternating : (n → R) [⋀^n]→ₗ[R] R).map_update_smul M j s u


theorem det_updateCol_smul (M : Matrix n n R) (j : n) (s : R) (u : n → R) :
    det (updateCol M j <| s • u) = s * det (updateCol M j u) := by
  /-
    n : Type u_2
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    R : Type v
    inst✝ : CommRing R
    M : Matrix n n R
    j : n
    s : R
    u : n → R
    ⊢ Eq (M.updateCol j (HSMul.hSMul s u)).det (HMul.hMul s (M.updateCol j u).det)
  -/
  rw [← det_transpose, ← updateRow_transpose, det_updateRow_smul]
  /-
    n : Type u_2
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    R : Type v
    inst✝ : CommRing R
    M : Matrix n n R
    j : n
    s : R
    u : n → R
    ⊢ Eq (HMul.hMul s (M.transpose.updateRow j u).det) (HMul.hMul s (M.updateCol j …
  -/
  simp [updateRow_transpose, det_transpose]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-12-11")] alias det_updateColumn_smul := det_updateCol_smul


theorem det_updateRow_smul_left (M : Matrix n n R) (j : n) (s : R) (u : n → R) :
    det (updateRow (s • M) j u) = s ^ (Fintype.card n - 1) * det (updateRow M j u) :=
  MultilinearMap.map_update_smul_left _ M j s u


@[deprecated (since := "2024-11-03")] alias det_updateRow_smul' := det_updateRow_smul_left


theorem det_updateCol_smul_left (M : Matrix n n R) (j : n) (s : R) (u : n → R) :
    det (updateCol (s • M) j u) = s ^ (Fintype.card n - 1) * det (updateCol M j u) := by
  /-
    n : Type u_2
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    R : Type v
    inst✝ : CommRing R
    M : Matrix n n R
    j : n
    s : R
    u : n → R
    ⊢ Eq ((HSMul.hSMul s M).updateCol j u).det (HMul.hMul (HPow.hPow s (HSub.hSub  …
  -/
  rw [← det_transpose, ← updateRow_transpose, transpose_smul, det_updateRow_smul_left]
  /-
    n : Type u_2
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    R : Type v
    inst✝ : CommRing R
    M : Matrix n n R
    j : n
    s : R
    u : n → R
    ⊢ Eq (HMul.hMul (HPow.hPow s (HSub.hSub (Fintype.card n) 1)) (M.transpose.upda …
  -/
  simp [updateRow_transpose, det_transpose]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-12-11")] alias det_updateColumn_smul' := det_updateCol_smul_left

@[deprecated (since := "2024-12-11")] alias det_updateColumn_smul_left := det_updateCol_smul_left


theorem det_updateRow_sum_aux (M : Matrix n n R) {j : n} (s : Finset n) (hj : j ∉ s) (c : n → R)
    (a : R) :
    (M.updateRow j (a • M j + ∑ k ∈ s, (c k) • M k)).det = a • M.det := by
  induction s using Finset.induction_on with
  | empty => rw [Finset.sum_empty, add_zero, smul_eq_mul, det_updateRow_smul, updateRow_eq_self]
  | @insert k _ hk h_ind =>
      have h : k ≠ j := fun h ↦ (h ▸ hj) (Finset.mem_insert_self _ _)
      rw [Finset.sum_insert hk, add_comm ((c k) • M k), ← add_assoc, det_updateRow_add,
        det_updateRow_smul, det_updateRow_eq_zero h, mul_zero, add_zero, h_ind]
      exact fun h ↦ hj (Finset.mem_insert_of_mem h)


/-- If we replace a row of a matrix by a linear combination of its rows, then the determinant is
multiplied by the coefficient of that row. -/
theorem det_updateRow_sum (A : Matrix n n R) (j : n) (c : n → R) :
    (A.updateRow j (∑ k, (c k) • A k)).det = (c j) • A.det := by
  /-
    n : Type u_2
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    R : Type v
    inst✝ : CommRing R
    A : Matrix n n R
    j : n
    c : n → R
    ⊢ Eq (A.updateRow j (Finset.univ.sum fun k => HSMul.hSMul (c k) (A k))).det (H …
  -/
  convert det_updateRow_sum_aux A (Finset.univ.erase j) (Finset.univ.not_mem_erase j) c (c j)
  /-
    case h.e'_2.h.e'_6.h.e'_7
    n : Type u_2
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    R : Type v
    inst✝ : CommRing R
    A : Matrix n n R
    j : n
    c : n → R
    ⊢ Eq (Finset.univ.sum fun k => HSMul.hSMul (c k) (A k)) (HAdd.hAdd (HSMul.hSMu …
  -/
  rw [← Finset.univ.add_sum_erase _ (Finset.mem_univ j)]
  /-
    🎉 no goals
  -/


/-- If we replace a column of a matrix by a linear combination of its columns, then the determinant
is multiplied by the coefficient of that column. -/
theorem det_updateCol_sum (A : Matrix n n R) (j : n) (c : n → R) :
    (A.updateCol j (fun k ↦ ∑ i, (c i) • A k i)).det = (c j) • A.det := by
  /-
    n : Type u_2
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    R : Type v
    inst✝ : CommRing R
    A : Matrix n n R
    j : n
    c : n → R
    ⊢ Eq (A.updateCol j fun k => Finset.univ.sum fun i => HSMul.hSMul (c i) (A k i …
  -/
  rw [← det_transpose, ← updateRow_transpose, ← det_transpose A]
  /-
    n : Type u_2
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    R : Type v
    inst✝ : CommRing R
    A : Matrix n n R
    j : n
    c : n → R
    ⊢ Eq (A.transpose.updateRow j fun k => Finset.univ.sum fun i => HSMul.hSMul (c …
  -/
  convert det_updateRow_sum A.transpose j c
  /-
    case h.e'_2.h.e'_6.h.e'_7.h
    n : Type u_2
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    R : Type v
    inst✝ : CommRing R
    A : Matrix n n R
    j : n
    c : n → R
    x✝ : n
    ⊢ Eq (Finset.univ.sum fun i => HSMul.hSMul (c i) (A x✝ i)) (Finset.univ.sum (f …
  -/
  simp only [smul_eq_mul, Finset.sum_apply, Pi.smul_apply, transpose_apply]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-12-11")] alias det_updateColumn_sum := det_updateCol_sum


theorem det_eq_of_eq_mul_det_one {A B : Matrix n n R} (C : Matrix n n R) (hC : det C = 1)
    (hA : A = B * C) : det A = det B :=
  calc
    det A = det (B * C) := congr_arg _ hA
    _ = det B * det C := det_mul _ _
                    /-
                      n : Type u_2
                      inst✝² : DecidableEq n
                      inst✝¹ : Fintype n
                      R : Type v
                      inst✝ : CommRing R
                      A B C : Matrix n n R
                      hC : Eq C.det 1
                      hA : Eq A (HMul.hMul B C)
                      ⊢ Eq (HMul.hMul B.det C.det) B.det
                    -/
    _ = det B := by rw [hC, mul_one]
                    /-
                      🎉 no goals
                    -/


theorem det_eq_of_eq_det_one_mul {A B : Matrix n n R} (C : Matrix n n R) (hC : det C = 1)
    (hA : A = C * B) : det A = det B :=
  calc
    det A = det (C * B) := congr_arg _ hA
    _ = det C * det B := det_mul _ _
                    /-
                      n : Type u_2
                      inst✝² : DecidableEq n
                      inst✝¹ : Fintype n
                      R : Type v
                      inst✝ : CommRing R
                      A B C : Matrix n n R
                      hC : Eq C.det 1
                      hA : Eq A (HMul.hMul C B)
                      ⊢ Eq (HMul.hMul C.det B.det) B.det
                    -/
    _ = det B := by rw [hC, one_mul]
                    /-
                      🎉 no goals
                    -/


theorem det_updateRow_add_self (A : Matrix n n R) {i j : n} (hij : i ≠ j) :
    det (updateRow A i (A i + A j)) = det A := by
  simp [det_updateRow_add,
    det_zero_of_row_eq hij (updateRow_self.trans (updateRow_ne hij.symm).symm)]


theorem det_updateCol_add_self (A : Matrix n n R) {i j : n} (hij : i ≠ j) :
    det (updateCol A i fun k => A k i + A k j) = det A := by
  /-
    n : Type u_2
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    R : Type v
    inst✝ : CommRing R
    A : Matrix n n R
    i j : n
    hij : Ne i j
    ⊢ Eq (A.updateCol i fun k => HAdd.hAdd (A k i) (A k j)).det A.det
  -/
  rw [← det_transpose, ← updateRow_transpose, ← det_transpose A]
  /-
    n : Type u_2
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    R : Type v
    inst✝ : CommRing R
    A : Matrix n n R
    i j : n
    hij : Ne i j
    ⊢ Eq (A.transpose.updateRow i fun k => HAdd.hAdd (A k i) (A k j)).det A.transp …
  -/
  exact det_updateRow_add_self Aᵀ hij
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-12-11")] alias det_updateColumn_add_self := det_updateCol_add_self


theorem det_updateRow_add_smul_self (A : Matrix n n R) {i j : n} (hij : i ≠ j) (c : R) :
    det (updateRow A i (A i + c • A j)) = det A := by
  simp [det_updateRow_add, det_updateRow_smul,
    det_zero_of_row_eq hij (updateRow_self.trans (updateRow_ne hij.symm).symm)]


theorem det_updateCol_add_smul_self (A : Matrix n n R) {i j : n} (hij : i ≠ j) (c : R) :
    det (updateCol A i fun k => A k i + c • A k j) = det A := by
  /-
    n : Type u_2
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    R : Type v
    inst✝ : CommRing R
    A : Matrix n n R
    i j : n
    hij : Ne i j
    c : R
    ⊢ Eq (A.updateCol i fun k => HAdd.hAdd (A k i) (HSMul.hSMul c (A k j))).det A. …
  -/
  rw [← det_transpose, ← updateRow_transpose, ← det_transpose A]
  /-
    n : Type u_2
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    R : Type v
    inst✝ : CommRing R
    A : Matrix n n R
    i j : n
    hij : Ne i j
    c : R
    ⊢ Eq (A.transpose.updateRow i fun k => HAdd.hAdd (A k i) (HSMul.hSMul c (A k j …
  -/
  exact det_updateRow_add_smul_self Aᵀ hij c
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-12-11")]
alias det_updateColumn_add_smul_self := det_updateCol_add_smul_self


theorem det_eq_of_forall_row_eq_smul_add_const_aux {A B : Matrix n n R} {s : Finset n} :
    ∀ (c : n → R) (_ : ∀ i, i ∉ s → c i = 0) (k : n) (_ : k ∉ s)
      (_ : ∀ i j, A i j = B i j + c i * B k j), det A = det B := by
  induction s using Finset.induction_on generalizing B with
  | empty =>
    rintro c hs k - A_eq
    have : ∀ i, c i = 0 := by
      intro i
      specialize hs i
      contrapose! hs
      simp [hs]
    congr
    ext i j
    rw [A_eq, this, zero_mul, add_zero]
  | @insert i s _hi ih =>
    intro c hs k hk A_eq
    have hAi : A i = B i + c i • B k := funext (A_eq i)
    rw [@ih (updateRow B i (A i)) (Function.update c i 0), hAi, det_updateRow_add_smul_self]
    · exact mt (fun h => show k ∈ insert i s from h ▸ Finset.mem_insert_self _ _) hk
    · intro i' hi'
      rw [Function.update_apply]
      split_ifs with hi'i
      · rfl
      · exact hs i' fun h => hi' ((Finset.mem_insert.mp h).resolve_left hi'i)
    · exact k
    · exact fun h => hk (Finset.mem_insert_of_mem h)
    · intro i' j'
      rw [updateRow_apply, Function.update_apply]
      split_ifs with hi'i
      · simp [hi'i]
      rw [A_eq, updateRow_ne fun h : k = i => hk <| h ▸ Finset.mem_insert_self k s]


/-- If you add multiples of row `B k` to other rows, the determinant doesn't change. -/
theorem det_eq_of_forall_row_eq_smul_add_const {A B : Matrix n n R} (c : n → R) (k : n)
    (hk : c k = 0) (A_eq : ∀ i j, A i j = B i j + c i * B k j) : det A = det B :=
  det_eq_of_forall_row_eq_smul_add_const_aux c
    (fun i =>
      not_imp_comm.mp fun hi =>
        Finset.mem_erase.mpr
          ⟨mt (fun h : i = k => show c i = 0 from h.symm ▸ hk) hi, Finset.mem_univ i⟩)
    k (Finset.not_mem_erase k Finset.univ) A_eq


theorem det_eq_of_forall_row_eq_smul_add_pred_aux {n : ℕ} (k : Fin (n + 1)) :
    ∀ (c : Fin n → R) (_hc : ∀ i : Fin n, k < i.succ → c i = 0)
      {M N : Matrix (Fin n.succ) (Fin n.succ) R} (_h0 : ∀ j, M 0 j = N 0 j)
      (_hsucc : ∀ (i : Fin n) (j), M i.succ j = N i.succ j + c i * M (Fin.castSucc i) j),
      det M = det N := by
  /-
    R : Type v
    inst✝ : CommRing R
    n : Nat
    k : Fin (HAdd.hAdd n 1)
    ⊢ ∀ (c : Fin n → R), (∀ (i : Fin n), LT.lt k i.succ → Eq (c i) 0) → ∀ {M N : M …
  -/
  refine Fin.induction ?_ (fun k ih => ?_) k <;> intro c hc M N h0 hsucc
    /-
      case refine_1
      R : Type v
      inst✝ : CommRing R
      n : Nat
      k : Fin (HAdd.hAdd n 1)
      c : Fin n → R
      hc : ∀ (i : Fin n), LT.lt 0 i.succ → Eq (c i) 0
      M N : Matrix (Fin n.succ) (Fin n.succ) R
      h0 : ∀ (j : Fin n.succ), Eq (M 0 j) (N 0 j)
      hsucc : ∀ (i : Fin n) (j : Fin n.succ), Eq (M i.succ j) (HAdd.hAdd (N i.succ j …
      ⊢ Eq M.det N.det
    -/
  · congr
    /-
      case refine_1.e_M
      R : Type v
      inst✝ : CommRing R
      n : Nat
      k : Fin (HAdd.hAdd n 1)
      c : Fin n → R
      hc : ∀ (i : Fin n), LT.lt 0 i.succ → Eq (c i) 0
      M N : Matrix (Fin n.succ) (Fin n.succ) R
      h0 : ∀ (j : Fin n.succ), Eq (M 0 j) (N 0 j)
      hsucc : ∀ (i : Fin n) (j : Fin n.succ), Eq (M i.succ j) (HAdd.hAdd (N i.succ j …
      ⊢ Eq M N
    -/
    ext i j
    /-
      case refine_1.e_M.a
      R : Type v
      inst✝ : CommRing R
      n : Nat
      k : Fin (HAdd.hAdd n 1)
      c : Fin n → R
      hc : ∀ (i : Fin n), LT.lt 0 i.succ → Eq (c i) 0
      M N : Matrix (Fin n.succ) (Fin n.succ) R
      h0 : ∀ (j : Fin n.succ), Eq (M 0 j) (N 0 j)
      hsucc : ∀ (i : Fin n) (j : Fin n.succ), Eq (M i.succ j) (HAdd.hAdd (N i.succ j …
      i j : Fin n.succ
      ⊢ Eq (M i j) (N i j)
    -/
    refine Fin.cases (h0 j) (fun i => ?_) i
    /-
      case refine_1.e_M.a
      R : Type v
      inst✝ : CommRing R
      n : Nat
      k : Fin (HAdd.hAdd n 1)
      c : Fin n → R
      hc : ∀ (i : Fin n), LT.lt 0 i.succ → Eq (c i) 0
      M N : Matrix (Fin n.succ) (Fin n.succ) R
      h0 : ∀ (j : Fin n.succ), Eq (M 0 j) (N 0 j)
      hsucc : ∀ (i : Fin n) (j : Fin n.succ), Eq (M i.succ j) (HAdd.hAdd (N i.succ j …
      i✝ j : Fin n.succ
      i : Fin n
      ⊢ Eq (M i.succ j) (N i.succ j)
    -/
    rw [hsucc, hc i (Fin.succ_pos _), zero_mul, add_zero]
    /-
      🎉 no goals
    -/
  /-
    case refine_2
    R : Type v
    inst✝ : CommRing R
    n : Nat
    k✝ : Fin (HAdd.hAdd n 1)
    k : Fin n
    ih : ∀ (c : Fin n → R), (∀ (i : Fin n), LT.lt k.castSucc i.succ → Eq (c i) 0)  …
    c : Fin n → R
    hc : ∀ (i : Fin n), LT.lt k.succ i.succ → Eq (c i) 0
    M N : Matrix (Fin n.succ) (Fin n.succ) R
    h0 : ∀ (j : Fin n.succ), Eq (M 0 j) (N 0 j)
    hsucc : ∀ (i : Fin n) (j : Fin n.succ), Eq (M i.succ j) (HAdd.hAdd (N i.succ j …
    ⊢ Eq M.det N.det
  -/
  set M' := updateRow M k.succ (N k.succ) with hM'
  have hM : M = updateRow M' k.succ (M' k.succ + c k • M (Fin.castSucc k)) := by
    ext i j
    by_cases hi : i = k.succ
    · simp [hi, hM', hsucc, updateRow_self]
    rw [updateRow_ne hi, hM', updateRow_ne hi]
  /-
    case refine_2
    R : Type v
    inst✝ : CommRing R
    n : Nat
    k✝ : Fin (HAdd.hAdd n 1)
    k : Fin n
    ih : ∀ (c : Fin n → R), (∀ (i : Fin n), LT.lt k.castSucc i.succ → Eq (c i) 0)  …
    c : Fin n → R
    hc : ∀ (i : Fin n), LT.lt k.succ i.succ → Eq (c i) 0
    M N : Matrix (Fin n.succ) (Fin n.succ) R
    h0 : ∀ (j : Fin n.succ), Eq (M 0 j) (N 0 j)
    hsucc : ∀ (i : Fin n) (j : Fin n.succ), Eq (M i.succ j) (HAdd.hAdd (N i.succ j …
    M' : Matrix (Fin n.succ) (Fin n.succ) R := M.updateRow k.succ (N k.succ)
    hM' : Eq M' (M.updateRow k.succ (N k.succ))
    hM : Eq M (M'.updateRow k.succ (HAdd.hAdd (M' k.succ) (HSMul.hSMul (c k) (M k. …
    ⊢ Eq M.det N.det
  -/
  have k_ne_succ : (Fin.castSucc k) ≠ k.succ := (Fin.castSucc_lt_succ k).ne
  /-
    case refine_2
    R : Type v
    inst✝ : CommRing R
    n : Nat
    k✝ : Fin (HAdd.hAdd n 1)
    k : Fin n
    ih : ∀ (c : Fin n → R), (∀ (i : Fin n), LT.lt k.castSucc i.succ → Eq (c i) 0)  …
    c : Fin n → R
    hc : ∀ (i : Fin n), LT.lt k.succ i.succ → Eq (c i) 0
    M N : Matrix (Fin n.succ) (Fin n.succ) R
    h0 : ∀ (j : Fin n.succ), Eq (M 0 j) (N 0 j)
    hsucc : ∀ (i : Fin n) (j : Fin n.succ), Eq (M i.succ j) (HAdd.hAdd (N i.succ j …
    M' : Matrix (Fin n.succ) (Fin n.succ) R := M.updateRow k.succ (N k.succ)
    hM' : Eq M' (M.updateRow k.succ (N k.succ))
    hM : Eq M (M'.updateRow k.succ (HAdd.hAdd (M' k.succ) (HSMul.hSMul (c k) (M k. …
    k_ne_succ : Ne k.castSucc k.succ
    ⊢ Eq M.det N.det
  -/
  have M_k : M (Fin.castSucc k) = M' (Fin.castSucc k) := (updateRow_ne k_ne_succ).symm
  /-
    case refine_2
    R : Type v
    inst✝ : CommRing R
    n : Nat
    k✝ : Fin (HAdd.hAdd n 1)
    k : Fin n
    ih : ∀ (c : Fin n → R), (∀ (i : Fin n), LT.lt k.castSucc i.succ → Eq (c i) 0)  …
    c : Fin n → R
    hc : ∀ (i : Fin n), LT.lt k.succ i.succ → Eq (c i) 0
    M N : Matrix (Fin n.succ) (Fin n.succ) R
    h0 : ∀ (j : Fin n.succ), Eq (M 0 j) (N 0 j)
    hsucc : ∀ (i : Fin n) (j : Fin n.succ), Eq (M i.succ j) (HAdd.hAdd (N i.succ j …
    M' : Matrix (Fin n.succ) (Fin n.succ) R := M.updateRow k.succ (N k.succ)
    hM' : Eq M' (M.updateRow k.succ (N k.succ))
    hM : Eq M (M'.updateRow k.succ (HAdd.hAdd (M' k.succ) (HSMul.hSMul (c k) (M k. …
    k_ne_succ : Ne k.castSucc k.succ
    M_k : Eq (M k.castSucc) (M' k.castSucc)
    ⊢ Eq M.det N.det
  -/
  rw [hM, M_k, det_updateRow_add_smul_self M' k_ne_succ.symm, ih (Function.update c k 0)]
    /-
      case refine_2._hc
      R : Type v
      inst✝ : CommRing R
      n : Nat
      k✝ : Fin (HAdd.hAdd n 1)
      k : Fin n
      ih : ∀ (c : Fin n → R), (∀ (i : Fin n), LT.lt k.castSucc i.succ → Eq (c i) 0)  …
      c : Fin n → R
      hc : ∀ (i : Fin n), LT.lt k.succ i.succ → Eq (c i) 0
      M N : Matrix (Fin n.succ) (Fin n.succ) R
      h0 : ∀ (j : Fin n.succ), Eq (M 0 j) (N 0 j)
      hsucc : ∀ (i : Fin n) (j : Fin n.succ), Eq (M i.succ j) (HAdd.hAdd (N i.succ j …
      M' : Matrix (Fin n.succ) (Fin n.succ) R := M.updateRow k.succ (N k.succ)
      hM' : Eq M' (M.updateRow k.succ (N k.succ))
      hM : Eq M (M'.updateRow k.succ (HAdd.hAdd (M' k.succ) (HSMul.hSMul (c k) (M k. …
      k_ne_succ : Ne k.castSucc k.succ
      M_k : Eq (M k.castSucc) (M' k.castSucc)
      ⊢ ∀ (i : Fin n), LT.lt k.castSucc i.succ → Eq (Function.update c k 0 i) 0
    -/
  · intro i hi
    /-
      case refine_2._hc
      R : Type v
      inst✝ : CommRing R
      n : Nat
      k✝ : Fin (HAdd.hAdd n 1)
      k : Fin n
      ih : ∀ (c : Fin n → R), (∀ (i : Fin n), LT.lt k.castSucc i.succ → Eq (c i) 0)  …
      c : Fin n → R
      hc : ∀ (i : Fin n), LT.lt k.succ i.succ → Eq (c i) 0
      M N : Matrix (Fin n.succ) (Fin n.succ) R
      h0 : ∀ (j : Fin n.succ), Eq (M 0 j) (N 0 j)
      hsucc : ∀ (i : Fin n) (j : Fin n.succ), Eq (M i.succ j) (HAdd.hAdd (N i.succ j …
      M' : Matrix (Fin n.succ) (Fin n.succ) R := M.updateRow k.succ (N k.succ)
      hM' : Eq M' (M.updateRow k.succ (N k.succ))
      hM : Eq M (M'.updateRow k.succ (HAdd.hAdd (M' k.succ) (HSMul.hSMul (c k) (M k. …
      k_ne_succ : Ne k.castSucc k.succ
      M_k : Eq (M k.castSucc) (M' k.castSucc)
      i : Fin n
      hi : LT.lt k.castSucc i.succ
      ⊢ Eq (Function.update c k 0 i) 0
    -/
    rw [Fin.lt_iff_val_lt_val, Fin.coe_castSucc, Fin.val_succ, Nat.lt_succ_iff] at hi
    /-
      case refine_2._hc
      R : Type v
      inst✝ : CommRing R
      n : Nat
      k✝ : Fin (HAdd.hAdd n 1)
      k : Fin n
      ih : ∀ (c : Fin n → R), (∀ (i : Fin n), LT.lt k.castSucc i.succ → Eq (c i) 0)  …
      c : Fin n → R
      hc : ∀ (i : Fin n), LT.lt k.succ i.succ → Eq (c i) 0
      M N : Matrix (Fin n.succ) (Fin n.succ) R
      h0 : ∀ (j : Fin n.succ), Eq (M 0 j) (N 0 j)
      hsucc : ∀ (i : Fin n) (j : Fin n.succ), Eq (M i.succ j) (HAdd.hAdd (N i.succ j …
      M' : Matrix (Fin n.succ) (Fin n.succ) R := M.updateRow k.succ (N k.succ)
      hM' : Eq M' (M.updateRow k.succ (N k.succ))
      hM : Eq M (M'.updateRow k.succ (HAdd.hAdd (M' k.succ) (HSMul.hSMul (c k) (M k. …
      k_ne_succ : Ne k.castSucc k.succ
      M_k : Eq (M k.castSucc) (M' k.castSucc)
      i : Fin n
      hi : LE.le ↑k ↑i
      ⊢ Eq (Function.update c k 0 i) 0
    -/
    rw [Function.update_apply]
    /-
      case refine_2._hc
      R : Type v
      inst✝ : CommRing R
      n : Nat
      k✝ : Fin (HAdd.hAdd n 1)
      k : Fin n
      ih : ∀ (c : Fin n → R), (∀ (i : Fin n), LT.lt k.castSucc i.succ → Eq (c i) 0)  …
      c : Fin n → R
      hc : ∀ (i : Fin n), LT.lt k.succ i.succ → Eq (c i) 0
      M N : Matrix (Fin n.succ) (Fin n.succ) R
      h0 : ∀ (j : Fin n.succ), Eq (M 0 j) (N 0 j)
      hsucc : ∀ (i : Fin n) (j : Fin n.succ), Eq (M i.succ j) (HAdd.hAdd (N i.succ j …
      M' : Matrix (Fin n.succ) (Fin n.succ) R := M.updateRow k.succ (N k.succ)
      hM' : Eq M' (M.updateRow k.succ (N k.succ))
      hM : Eq M (M'.updateRow k.succ (HAdd.hAdd (M' k.succ) (HSMul.hSMul (c k) (M k. …
      k_ne_succ : Ne k.castSucc k.succ
      M_k : Eq (M k.castSucc) (M' k.castSucc)
      i : Fin n
      hi : LE.le ↑k ↑i
      ⊢ Eq (ite (Eq i k) 0 (c i)) 0
    -/
    split_ifs with hik
      /-
        case pos
        R : Type v
        inst✝ : CommRing R
        n : Nat
        k✝ : Fin (HAdd.hAdd n 1)
        k : Fin n
        ih : ∀ (c : Fin n → R), (∀ (i : Fin n), LT.lt k.castSucc i.succ → Eq (c i) 0)  …
        c : Fin n → R
        hc : ∀ (i : Fin n), LT.lt k.succ i.succ → Eq (c i) 0
        M N : Matrix (Fin n.succ) (Fin n.succ) R
        h0 : ∀ (j : Fin n.succ), Eq (M 0 j) (N 0 j)
        hsucc : ∀ (i : Fin n) (j : Fin n.succ), Eq (M i.succ j) (HAdd.hAdd (N i.succ j …
        M' : Matrix (Fin n.succ) (Fin n.succ) R := M.updateRow k.succ (N k.succ)
        hM' : Eq M' (M.updateRow k.succ (N k.succ))
        hM : Eq M (M'.updateRow k.succ (HAdd.hAdd (M' k.succ) (HSMul.hSMul (c k) (M k. …
        k_ne_succ : Ne k.castSucc k.succ
        M_k : Eq (M k.castSucc) (M' k.castSucc)
        i : Fin n
        hi : LE.le ↑k ↑i
        hik : Eq i k
        ⊢ Eq 0 0
      -/
    · rfl
      /-
        🎉 no goals
      -/
    /-
      case neg
      R : Type v
      inst✝ : CommRing R
      n : Nat
      k✝ : Fin (HAdd.hAdd n 1)
      k : Fin n
      ih : ∀ (c : Fin n → R), (∀ (i : Fin n), LT.lt k.castSucc i.succ → Eq (c i) 0)  …
      c : Fin n → R
      hc : ∀ (i : Fin n), LT.lt k.succ i.succ → Eq (c i) 0
      M N : Matrix (Fin n.succ) (Fin n.succ) R
      h0 : ∀ (j : Fin n.succ), Eq (M 0 j) (N 0 j)
      hsucc : ∀ (i : Fin n) (j : Fin n.succ), Eq (M i.succ j) (HAdd.hAdd (N i.succ j …
      M' : Matrix (Fin n.succ) (Fin n.succ) R := M.updateRow k.succ (N k.succ)
      hM' : Eq M' (M.updateRow k.succ (N k.succ))
      hM : Eq M (M'.updateRow k.succ (HAdd.hAdd (M' k.succ) (HSMul.hSMul (c k) (M k. …
      k_ne_succ : Ne k.castSucc k.succ
      M_k : Eq (M k.castSucc) (M' k.castSucc)
      i : Fin n
      hi : LE.le ↑k ↑i
      hik : Not (Eq i k)
      ⊢ Eq (c i) 0
    -/
    exact hc _ (Fin.succ_lt_succ_iff.mpr (lt_of_le_of_ne hi (Ne.symm hik)))
    /-
      🎉 no goals
    -/
    /-
      case refine_2._h0
      R : Type v
      inst✝ : CommRing R
      n : Nat
      k✝ : Fin (HAdd.hAdd n 1)
      k : Fin n
      ih : ∀ (c : Fin n → R), (∀ (i : Fin n), LT.lt k.castSucc i.succ → Eq (c i) 0)  …
      c : Fin n → R
      hc : ∀ (i : Fin n), LT.lt k.succ i.succ → Eq (c i) 0
      M N : Matrix (Fin n.succ) (Fin n.succ) R
      h0 : ∀ (j : Fin n.succ), Eq (M 0 j) (N 0 j)
      hsucc : ∀ (i : Fin n) (j : Fin n.succ), Eq (M i.succ j) (HAdd.hAdd (N i.succ j …
      M' : Matrix (Fin n.succ) (Fin n.succ) R := M.updateRow k.succ (N k.succ)
      hM' : Eq M' (M.updateRow k.succ (N k.succ))
      hM : Eq M (M'.updateRow k.succ (HAdd.hAdd (M' k.succ) (HSMul.hSMul (c k) (M k. …
      k_ne_succ : Ne k.castSucc k.succ
      M_k : Eq (M k.castSucc) (M' k.castSucc)
      ⊢ ∀ (j : Fin n.succ), Eq (M' 0 j) (N 0 j)
    -/
  · rwa [hM', updateRow_ne (Fin.succ_ne_zero _).symm]
    /-
      🎉 no goals
    -/
  /-
    case refine_2._hsucc
    R : Type v
    inst✝ : CommRing R
    n : Nat
    k✝ : Fin (HAdd.hAdd n 1)
    k : Fin n
    ih : ∀ (c : Fin n → R), (∀ (i : Fin n), LT.lt k.castSucc i.succ → Eq (c i) 0)  …
    c : Fin n → R
    hc : ∀ (i : Fin n), LT.lt k.succ i.succ → Eq (c i) 0
    M N : Matrix (Fin n.succ) (Fin n.succ) R
    h0 : ∀ (j : Fin n.succ), Eq (M 0 j) (N 0 j)
    hsucc : ∀ (i : Fin n) (j : Fin n.succ), Eq (M i.succ j) (HAdd.hAdd (N i.succ j …
    M' : Matrix (Fin n.succ) (Fin n.succ) R := M.updateRow k.succ (N k.succ)
    hM' : Eq M' (M.updateRow k.succ (N k.succ))
    hM : Eq M (M'.updateRow k.succ (HAdd.hAdd (M' k.succ) (HSMul.hSMul (c k) (M k. …
    k_ne_succ : Ne k.castSucc k.succ
    M_k : Eq (M k.castSucc) (M' k.castSucc)
    ⊢ ∀ (i : Fin n) (j : Fin n.succ), Eq (M' i.succ j) (HAdd.hAdd (N i.succ j) (HM …
  -/
  intro i j
  /-
    case refine_2._hsucc
    R : Type v
    inst✝ : CommRing R
    n : Nat
    k✝ : Fin (HAdd.hAdd n 1)
    k : Fin n
    ih : ∀ (c : Fin n → R), (∀ (i : Fin n), LT.lt k.castSucc i.succ → Eq (c i) 0)  …
    c : Fin n → R
    hc : ∀ (i : Fin n), LT.lt k.succ i.succ → Eq (c i) 0
    M N : Matrix (Fin n.succ) (Fin n.succ) R
    h0 : ∀ (j : Fin n.succ), Eq (M 0 j) (N 0 j)
    hsucc : ∀ (i : Fin n) (j : Fin n.succ), Eq (M i.succ j) (HAdd.hAdd (N i.succ j …
    M' : Matrix (Fin n.succ) (Fin n.succ) R := M.updateRow k.succ (N k.succ)
    hM' : Eq M' (M.updateRow k.succ (N k.succ))
    hM : Eq M (M'.updateRow k.succ (HAdd.hAdd (M' k.succ) (HSMul.hSMul (c k) (M k. …
    k_ne_succ : Ne k.castSucc k.succ
    M_k : Eq (M k.castSucc) (M' k.castSucc)
    i : Fin n
    j : Fin n.succ
    ⊢ Eq (M' i.succ j) (HAdd.hAdd (N i.succ j) (HMul.hMul (Function.update c k 0 i …
  -/
  rw [Function.update_apply]
  /-
    case refine_2._hsucc
    R : Type v
    inst✝ : CommRing R
    n : Nat
    k✝ : Fin (HAdd.hAdd n 1)
    k : Fin n
    ih : ∀ (c : Fin n → R), (∀ (i : Fin n), LT.lt k.castSucc i.succ → Eq (c i) 0)  …
    c : Fin n → R
    hc : ∀ (i : Fin n), LT.lt k.succ i.succ → Eq (c i) 0
    M N : Matrix (Fin n.succ) (Fin n.succ) R
    h0 : ∀ (j : Fin n.succ), Eq (M 0 j) (N 0 j)
    hsucc : ∀ (i : Fin n) (j : Fin n.succ), Eq (M i.succ j) (HAdd.hAdd (N i.succ j …
    M' : Matrix (Fin n.succ) (Fin n.succ) R := M.updateRow k.succ (N k.succ)
    hM' : Eq M' (M.updateRow k.succ (N k.succ))
    hM : Eq M (M'.updateRow k.succ (HAdd.hAdd (M' k.succ) (HSMul.hSMul (c k) (M k. …
    k_ne_succ : Ne k.castSucc k.succ
    M_k : Eq (M k.castSucc) (M' k.castSucc)
    i : Fin n
    j : Fin n.succ
    ⊢ Eq (M' i.succ j) (HAdd.hAdd (N i.succ j) (HMul.hMul (ite (Eq i k) 0 (c i)) ( …
  -/
  split_ifs with hik
    /-
      case pos
      R : Type v
      inst✝ : CommRing R
      n : Nat
      k✝ : Fin (HAdd.hAdd n 1)
      k : Fin n
      ih : ∀ (c : Fin n → R), (∀ (i : Fin n), LT.lt k.castSucc i.succ → Eq (c i) 0)  …
      c : Fin n → R
      hc : ∀ (i : Fin n), LT.lt k.succ i.succ → Eq (c i) 0
      M N : Matrix (Fin n.succ) (Fin n.succ) R
      h0 : ∀ (j : Fin n.succ), Eq (M 0 j) (N 0 j)
      hsucc : ∀ (i : Fin n) (j : Fin n.succ), Eq (M i.succ j) (HAdd.hAdd (N i.succ j …
      M' : Matrix (Fin n.succ) (Fin n.succ) R := M.updateRow k.succ (N k.succ)
      hM' : Eq M' (M.updateRow k.succ (N k.succ))
      hM : Eq M (M'.updateRow k.succ (HAdd.hAdd (M' k.succ) (HSMul.hSMul (c k) (M k. …
      k_ne_succ : Ne k.castSucc k.succ
      M_k : Eq (M k.castSucc) (M' k.castSucc)
      i : Fin n
      j : Fin n.succ
      hik : Eq i k
      ⊢ Eq (M' i.succ j) (HAdd.hAdd (N i.succ j) (HMul.hMul 0 (M' i.castSucc j)))
    -/
  · rw [zero_mul, add_zero, hM', hik, updateRow_self]
    /-
      🎉 no goals
    -/
  /-
    case neg
    R : Type v
    inst✝ : CommRing R
    n : Nat
    k✝ : Fin (HAdd.hAdd n 1)
    k : Fin n
    ih : ∀ (c : Fin n → R), (∀ (i : Fin n), LT.lt k.castSucc i.succ → Eq (c i) 0)  …
    c : Fin n → R
    hc : ∀ (i : Fin n), LT.lt k.succ i.succ → Eq (c i) 0
    M N : Matrix (Fin n.succ) (Fin n.succ) R
    h0 : ∀ (j : Fin n.succ), Eq (M 0 j) (N 0 j)
    hsucc : ∀ (i : Fin n) (j : Fin n.succ), Eq (M i.succ j) (HAdd.hAdd (N i.succ j …
    M' : Matrix (Fin n.succ) (Fin n.succ) R := M.updateRow k.succ (N k.succ)
    hM' : Eq M' (M.updateRow k.succ (N k.succ))
    hM : Eq M (M'.updateRow k.succ (HAdd.hAdd (M' k.succ) (HSMul.hSMul (c k) (M k. …
    k_ne_succ : Ne k.castSucc k.succ
    M_k : Eq (M k.castSucc) (M' k.castSucc)
    i : Fin n
    j : Fin n.succ
    hik : Not (Eq i k)
    ⊢ Eq (M' i.succ j) (HAdd.hAdd (N i.succ j) (HMul.hMul (c i) (M' i.castSucc j)))
  -/
  rw [hM', updateRow_ne ((Fin.succ_injective _).ne hik), hsucc]
  /-
    case neg
    R : Type v
    inst✝ : CommRing R
    n : Nat
    k✝ : Fin (HAdd.hAdd n 1)
    k : Fin n
    ih : ∀ (c : Fin n → R), (∀ (i : Fin n), LT.lt k.castSucc i.succ → Eq (c i) 0)  …
    c : Fin n → R
    hc : ∀ (i : Fin n), LT.lt k.succ i.succ → Eq (c i) 0
    M N : Matrix (Fin n.succ) (Fin n.succ) R
    h0 : ∀ (j : Fin n.succ), Eq (M 0 j) (N 0 j)
    hsucc : ∀ (i : Fin n) (j : Fin n.succ), Eq (M i.succ j) (HAdd.hAdd (N i.succ j …
    M' : Matrix (Fin n.succ) (Fin n.succ) R := M.updateRow k.succ (N k.succ)
    hM' : Eq M' (M.updateRow k.succ (N k.succ))
    hM : Eq M (M'.updateRow k.succ (HAdd.hAdd (M' k.succ) (HSMul.hSMul (c k) (M k. …
    k_ne_succ : Ne k.castSucc k.succ
    M_k : Eq (M k.castSucc) (M' k.castSucc)
    i : Fin n
    j : Fin n.succ
    hik : Not (Eq i k)
    ⊢ Eq (HAdd.hAdd (N i.succ j) (HMul.hMul (c i) (M i.castSucc j))) (HAdd.hAdd (N …
  -/
  by_cases hik2 : k < i
    /-
      case pos
      R : Type v
      inst✝ : CommRing R
      n : Nat
      k✝ : Fin (HAdd.hAdd n 1)
      k : Fin n
      ih : ∀ (c : Fin n → R), (∀ (i : Fin n), LT.lt k.castSucc i.succ → Eq (c i) 0)  …
      c : Fin n → R
      hc : ∀ (i : Fin n), LT.lt k.succ i.succ → Eq (c i) 0
      M N : Matrix (Fin n.succ) (Fin n.succ) R
      h0 : ∀ (j : Fin n.succ), Eq (M 0 j) (N 0 j)
      hsucc : ∀ (i : Fin n) (j : Fin n.succ), Eq (M i.succ j) (HAdd.hAdd (N i.succ j …
      M' : Matrix (Fin n.succ) (Fin n.succ) R := M.updateRow k.succ (N k.succ)
      hM' : Eq M' (M.updateRow k.succ (N k.succ))
      hM : Eq M (M'.updateRow k.succ (HAdd.hAdd (M' k.succ) (HSMul.hSMul (c k) (M k. …
      k_ne_succ : Ne k.castSucc k.succ
      M_k : Eq (M k.castSucc) (M' k.castSucc)
      i : Fin n
      j : Fin n.succ
      hik : Not (Eq i k)
      hik2 : LT.lt k i
      ⊢ Eq (HAdd.hAdd (N i.succ j) (HMul.hMul (c i) (M i.castSucc j))) (HAdd.hAdd (N …
    -/
  · simp [hc i (Fin.succ_lt_succ_iff.mpr hik2)]
    /-
      🎉 no goals
    -/
  /-
    case neg
    R : Type v
    inst✝ : CommRing R
    n : Nat
    k✝ : Fin (HAdd.hAdd n 1)
    k : Fin n
    ih : ∀ (c : Fin n → R), (∀ (i : Fin n), LT.lt k.castSucc i.succ → Eq (c i) 0)  …
    c : Fin n → R
    hc : ∀ (i : Fin n), LT.lt k.succ i.succ → Eq (c i) 0
    M N : Matrix (Fin n.succ) (Fin n.succ) R
    h0 : ∀ (j : Fin n.succ), Eq (M 0 j) (N 0 j)
    hsucc : ∀ (i : Fin n) (j : Fin n.succ), Eq (M i.succ j) (HAdd.hAdd (N i.succ j …
    M' : Matrix (Fin n.succ) (Fin n.succ) R := M.updateRow k.succ (N k.succ)
    hM' : Eq M' (M.updateRow k.succ (N k.succ))
    hM : Eq M (M'.updateRow k.succ (HAdd.hAdd (M' k.succ) (HSMul.hSMul (c k) (M k. …
    k_ne_succ : Ne k.castSucc k.succ
    M_k : Eq (M k.castSucc) (M' k.castSucc)
    i : Fin n
    j : Fin n.succ
    hik : Not (Eq i k)
    hik2 : Not (LT.lt k i)
    ⊢ Eq (HAdd.hAdd (N i.succ j) (HMul.hMul (c i) (M i.castSucc j))) (HAdd.hAdd (N …
  -/
  rw [updateRow_ne]
  /-
    case neg
    R : Type v
    inst✝ : CommRing R
    n : Nat
    k✝ : Fin (HAdd.hAdd n 1)
    k : Fin n
    ih : ∀ (c : Fin n → R), (∀ (i : Fin n), LT.lt k.castSucc i.succ → Eq (c i) 0)  …
    c : Fin n → R
    hc : ∀ (i : Fin n), LT.lt k.succ i.succ → Eq (c i) 0
    M N : Matrix (Fin n.succ) (Fin n.succ) R
    h0 : ∀ (j : Fin n.succ), Eq (M 0 j) (N 0 j)
    hsucc : ∀ (i : Fin n) (j : Fin n.succ), Eq (M i.succ j) (HAdd.hAdd (N i.succ j …
    M' : Matrix (Fin n.succ) (Fin n.succ) R := M.updateRow k.succ (N k.succ)
    hM' : Eq M' (M.updateRow k.succ (N k.succ))
    hM : Eq M (M'.updateRow k.succ (HAdd.hAdd (M' k.succ) (HSMul.hSMul (c k) (M k. …
    k_ne_succ : Ne k.castSucc k.succ
    M_k : Eq (M k.castSucc) (M' k.castSucc)
    i : Fin n
    j : Fin n.succ
    hik : Not (Eq i k)
    hik2 : Not (LT.lt k i)
    ⊢ Ne i.castSucc k.succ
  -/
  apply ne_of_lt
  /-
    case neg.h
    R : Type v
    inst✝ : CommRing R
    n : Nat
    k✝ : Fin (HAdd.hAdd n 1)
    k : Fin n
    ih : ∀ (c : Fin n → R), (∀ (i : Fin n), LT.lt k.castSucc i.succ → Eq (c i) 0)  …
    c : Fin n → R
    hc : ∀ (i : Fin n), LT.lt k.succ i.succ → Eq (c i) 0
    M N : Matrix (Fin n.succ) (Fin n.succ) R
    h0 : ∀ (j : Fin n.succ), Eq (M 0 j) (N 0 j)
    hsucc : ∀ (i : Fin n) (j : Fin n.succ), Eq (M i.succ j) (HAdd.hAdd (N i.succ j …
    M' : Matrix (Fin n.succ) (Fin n.succ) R := M.updateRow k.succ (N k.succ)
    hM' : Eq M' (M.updateRow k.succ (N k.succ))
    hM : Eq M (M'.updateRow k.succ (HAdd.hAdd (M' k.succ) (HSMul.hSMul (c k) (M k. …
    k_ne_succ : Ne k.castSucc k.succ
    M_k : Eq (M k.castSucc) (M' k.castSucc)
    i : Fin n
    j : Fin n.succ
    hik : Not (Eq i k)
    hik2 : Not (LT.lt k i)
    ⊢ LT.lt i.castSucc k.succ
  -/
  rwa [Fin.lt_iff_val_lt_val, Fin.coe_castSucc, Fin.val_succ, Nat.lt_succ_iff, ← not_lt]
  /-
    🎉 no goals
  -/


/-- If you add multiples of previous rows to the next row, the determinant doesn't change. -/
theorem det_eq_of_forall_row_eq_smul_add_pred {n : ℕ} {A B : Matrix (Fin (n + 1)) (Fin (n + 1)) R}
    (c : Fin n → R) (A_zero : ∀ j, A 0 j = B 0 j)
    (A_succ : ∀ (i : Fin n) (j), A i.succ j = B i.succ j + c i * A (Fin.castSucc i) j) :
    det A = det B :=
  det_eq_of_forall_row_eq_smul_add_pred_aux (Fin.last _) c
    (fun _ hi => absurd hi (not_lt_of_ge (Fin.le_last _))) A_zero A_succ


/-- If you add multiples of previous columns to the next columns, the determinant doesn't change. -/
theorem det_eq_of_forall_col_eq_smul_add_pred {n : ℕ} {A B : Matrix (Fin (n + 1)) (Fin (n + 1)) R}
    (c : Fin n → R) (A_zero : ∀ i, A i 0 = B i 0)
    (A_succ : ∀ (i) (j : Fin n), A i j.succ = B i j.succ + c j * A i (Fin.castSucc j)) :
    det A = det B := by
  /-
    R : Type v
    inst✝ : CommRing R
    n : Nat
    A B : Matrix (Fin (HAdd.hAdd n 1)) (Fin (HAdd.hAdd n 1)) R
    c : Fin n → R
    A_zero : ∀ (i : Fin (HAdd.hAdd n 1)), Eq (A i 0) (B i 0)
    A_succ : ∀ (i : Fin (HAdd.hAdd n 1)) (j : Fin n), Eq (A i j.succ) (HAdd.hAdd ( …
    ⊢ Eq A.det B.det
  -/
  rw [← det_transpose A, ← det_transpose B]
  /-
    R : Type v
    inst✝ : CommRing R
    n : Nat
    A B : Matrix (Fin (HAdd.hAdd n 1)) (Fin (HAdd.hAdd n 1)) R
    c : Fin n → R
    A_zero : ∀ (i : Fin (HAdd.hAdd n 1)), Eq (A i 0) (B i 0)
    A_succ : ∀ (i : Fin (HAdd.hAdd n 1)) (j : Fin n), Eq (A i j.succ) (HAdd.hAdd ( …
    ⊢ Eq A.transpose.det B.transpose.det
  -/
  exact det_eq_of_forall_row_eq_smul_add_pred c A_zero fun i j => A_succ j i
  /-
    🎉 no goals
  -/


@[simp]
theorem det_blockDiagonal {o : Type*} [Fintype o] [DecidableEq o] (M : o → Matrix n n R) :
    (blockDiagonal M).det = ∏ k, (M k).det := by
  -- Rewrite the determinants as a sum over permutations.
  /-
    n : Type u_2
    inst✝⁴ : DecidableEq n
    inst✝³ : Fintype n
    R : Type v
    inst✝² : CommRing R
    o : Type u_3
    inst✝¹ : Fintype o
    inst✝ : DecidableEq o
    M : o → Matrix n n R
    ⊢ Eq (Matrix.blockDiagonal M).det (Finset.univ.prod fun k => (M k).det)
  -/
  simp_rw [det_apply']
  -- The right hand side is a product of sums, rewrite it as a sum of products.
  /-
    n : Type u_2
    inst✝⁴ : DecidableEq n
    inst✝³ : Fintype n
    R : Type v
    inst✝² : CommRing R
    o : Type u_3
    inst✝¹ : Fintype o
    inst✝ : DecidableEq o
    M : o → Matrix n n R
    ⊢ Eq (Finset.univ.sum fun σ => HMul.hMul (↑↑(Equiv.Perm.sign σ)) (Finset.univ. …
  -/
  rw [Finset.prod_sum]
  /-
    n : Type u_2
    inst✝⁴ : DecidableEq n
    inst✝³ : Fintype n
    R : Type v
    inst✝² : CommRing R
    o : Type u_3
    inst✝¹ : Fintype o
    inst✝ : DecidableEq o
    M : o → Matrix n n R
    ⊢ Eq (Finset.univ.sum fun σ => HMul.hMul (↑↑(Equiv.Perm.sign σ)) (Finset.univ. …
  -/
  simp_rw [Finset.prod_attach_univ, Finset.univ_pi_univ]
  -- We claim that the only permutations contributing to the sum are those that
  -- preserve their second component.
  /-
    n : Type u_2
    inst✝⁴ : DecidableEq n
    inst✝³ : Fintype n
    R : Type v
    inst✝² : CommRing R
    o : Type u_3
    inst✝¹ : Fintype o
    inst✝ : DecidableEq o
    M : o → Matrix n n R
    ⊢ Eq (Finset.univ.sum fun σ => HMul.hMul (↑↑(Equiv.Perm.sign σ)) (Finset.univ. …
  -/
  let preserving_snd : Finset (Equiv.Perm (n × o)) := {σ | ∀ x, (σ x).snd = x.snd}
  have mem_preserving_snd :
    ∀ {σ : Equiv.Perm (n × o)}, σ ∈ preserving_snd ↔ ∀ x, (σ x).snd = x.snd := fun {σ} =>
    Finset.mem_filter.trans ⟨fun h => h.2, fun h => ⟨Finset.mem_univ _, h⟩⟩
  /-
    n : Type u_2
    inst✝⁴ : DecidableEq n
    inst✝³ : Fintype n
    R : Type v
    inst✝² : CommRing R
    o : Type u_3
    inst✝¹ : Fintype o
    inst✝ : DecidableEq o
    M : o → Matrix n n R
    preserving_snd : Finset (Equiv.Perm (Prod n o)) := Finset.filter (fun σ => ∀ ( …
    mem_preserving_snd : ∀ {σ : Equiv.Perm (Prod n o)}, Iff (Membership.mem preser …
    ⊢ Eq (Finset.univ.sum fun σ => HMul.hMul (↑↑(Equiv.Perm.sign σ)) (Finset.univ. …
  -/
  rw [← Finset.sum_subset (Finset.subset_univ preserving_snd) _]
  -- And that these are in bijection with `o → Equiv.Perm m`.
    /-
      n : Type u_2
      inst✝⁴ : DecidableEq n
      inst✝³ : Fintype n
      R : Type v
      inst✝² : CommRing R
      o : Type u_3
      inst✝¹ : Fintype o
      inst✝ : DecidableEq o
      M : o → Matrix n n R
      preserving_snd : Finset (Equiv.Perm (Prod n o)) := Finset.filter (fun σ => ∀ ( …
      mem_preserving_snd : ∀ {σ : Equiv.Perm (Prod n o)}, Iff (Membership.mem preser …
      ⊢ Eq (preserving_snd.sum fun x => HMul.hMul (↑↑(Equiv.Perm.sign x)) (Finset.un …
    -/
  · refine (Finset.sum_bij (fun σ _ => prodCongrLeft fun k ↦ σ k (mem_univ k)) ?_ ?_ ?_ ?_).symm
      /-
        case refine_1
        n : Type u_2
        inst✝⁴ : DecidableEq n
        inst✝³ : Fintype n
        R : Type v
        inst✝² : CommRing R
        o : Type u_3
        inst✝¹ : Fintype o
        inst✝ : DecidableEq o
        M : o → Matrix n n R
        preserving_snd : Finset (Equiv.Perm (Prod n o)) := Finset.filter (fun σ => ∀ ( …
        mem_preserving_snd : ∀ {σ : Equiv.Perm (Prod n o)}, Iff (Membership.mem preser …
        ⊢ ∀ (a : (a : o) → Membership.mem Finset.univ a → Equiv.Perm n) (ha : Membersh …
      -/
    · intro σ _
      /-
        case refine_1
        n : Type u_2
        inst✝⁴ : DecidableEq n
        inst✝³ : Fintype n
        R : Type v
        inst✝² : CommRing R
        o : Type u_3
        inst✝¹ : Fintype o
        inst✝ : DecidableEq o
        M : o → Matrix n n R
        preserving_snd : Finset (Equiv.Perm (Prod n o)) := Finset.filter (fun σ => ∀ ( …
        mem_preserving_snd : ∀ {σ : Equiv.Perm (Prod n o)}, Iff (Membership.mem preser …
        σ : (a : o) → Membership.mem Finset.univ a → Equiv.Perm n
        ha✝ : Membership.mem Finset.univ σ
        ⊢ Membership.mem preserving_snd ((fun σ x => Equiv.prodCongrLeft fun k => σ k  …
      -/
      rw [mem_preserving_snd]
      /-
        case refine_1
        n : Type u_2
        inst✝⁴ : DecidableEq n
        inst✝³ : Fintype n
        R : Type v
        inst✝² : CommRing R
        o : Type u_3
        inst✝¹ : Fintype o
        inst✝ : DecidableEq o
        M : o → Matrix n n R
        preserving_snd : Finset (Equiv.Perm (Prod n o)) := Finset.filter (fun σ => ∀ ( …
        mem_preserving_snd : ∀ {σ : Equiv.Perm (Prod n o)}, Iff (Membership.mem preser …
        σ : (a : o) → Membership.mem Finset.univ a → Equiv.Perm n
        ha✝ : Membership.mem Finset.univ σ
        ⊢ ∀ (x : Prod n o), Eq (((fun σ x => Equiv.prodCongrLeft fun k => σ k ⋯) σ ha✝ …
      -/
      rintro ⟨-, x⟩
      /-
        case refine_1.mk
        n : Type u_2
        inst✝⁴ : DecidableEq n
        inst✝³ : Fintype n
        R : Type v
        inst✝² : CommRing R
        o : Type u_3
        inst✝¹ : Fintype o
        inst✝ : DecidableEq o
        M : o → Matrix n n R
        preserving_snd : Finset (Equiv.Perm (Prod n o)) := Finset.filter (fun σ => ∀ ( …
        mem_preserving_snd : ∀ {σ : Equiv.Perm (Prod n o)}, Iff (Membership.mem preser …
        σ : (a : o) → Membership.mem Finset.univ a → Equiv.Perm n
        ha✝ : Membership.mem Finset.univ σ
        fst✝ : n
        x : o
        ⊢ Eq (((fun σ x => Equiv.prodCongrLeft fun k => σ k ⋯) σ ha✝) { fst := fst✝, s …
      -/
      simp only [prodCongrLeft_apply]
      /-
        🎉 no goals
      -/
      /-
        case refine_2
        n : Type u_2
        inst✝⁴ : DecidableEq n
        inst✝³ : Fintype n
        R : Type v
        inst✝² : CommRing R
        o : Type u_3
        inst✝¹ : Fintype o
        inst✝ : DecidableEq o
        M : o → Matrix n n R
        preserving_snd : Finset (Equiv.Perm (Prod n o)) := Finset.filter (fun σ => ∀ ( …
        mem_preserving_snd : ∀ {σ : Equiv.Perm (Prod n o)}, Iff (Membership.mem preser …
        ⊢ ∀ (a₁ : (a : o) → Membership.mem Finset.univ a → Equiv.Perm n) (ha₁ : Member …
      -/
    · intro σ _ σ' _ eq
      /-
        case refine_2
        n : Type u_2
        inst✝⁴ : DecidableEq n
        inst✝³ : Fintype n
        R : Type v
        inst✝² : CommRing R
        o : Type u_3
        inst✝¹ : Fintype o
        inst✝ : DecidableEq o
        M : o → Matrix n n R
        preserving_snd : Finset (Equiv.Perm (Prod n o)) := Finset.filter (fun σ => ∀ ( …
        mem_preserving_snd : ∀ {σ : Equiv.Perm (Prod n o)}, Iff (Membership.mem preser …
        σ : (a : o) → Membership.mem Finset.univ a → Equiv.Perm n
        ha₁✝ : Membership.mem Finset.univ σ
        σ' : (a : o) → Membership.mem Finset.univ a → Equiv.Perm n
        ha₂✝ : Membership.mem Finset.univ σ'
        eq : Eq ((fun σ x => Equiv.prodCongrLeft fun k => σ k ⋯) σ ha₁✝) ((fun σ x =>  …
        ⊢ Eq σ σ'
      -/
      ext x hx k
      /-
        case refine_2.h.h.H
        n : Type u_2
        inst✝⁴ : DecidableEq n
        inst✝³ : Fintype n
        R : Type v
        inst✝² : CommRing R
        o : Type u_3
        inst✝¹ : Fintype o
        inst✝ : DecidableEq o
        M : o → Matrix n n R
        preserving_snd : Finset (Equiv.Perm (Prod n o)) := Finset.filter (fun σ => ∀ ( …
        mem_preserving_snd : ∀ {σ : Equiv.Perm (Prod n o)}, Iff (Membership.mem preser …
        σ : (a : o) → Membership.mem Finset.univ a → Equiv.Perm n
        ha₁✝ : Membership.mem Finset.univ σ
        σ' : (a : o) → Membership.mem Finset.univ a → Equiv.Perm n
        ha₂✝ : Membership.mem Finset.univ σ'
        eq : Eq ((fun σ x => Equiv.prodCongrLeft fun k => σ k ⋯) σ ha₁✝) ((fun σ x =>  …
        x : o
        hx : Membership.mem Finset.univ x
        k : n
        ⊢ Eq ((σ x hx) k) ((σ' x hx) k)
      -/
      simp only at eq
      have :
        ∀ k x,
          prodCongrLeft (fun k => σ k (Finset.mem_univ _)) (k, x) =
            prodCongrLeft (fun k => σ' k (Finset.mem_univ _)) (k, x) :=
        fun k x => by rw [eq]
      /-
        case refine_2.h.h.H
        n : Type u_2
        inst✝⁴ : DecidableEq n
        inst✝³ : Fintype n
        R : Type v
        inst✝² : CommRing R
        o : Type u_3
        inst✝¹ : Fintype o
        inst✝ : DecidableEq o
        M : o → Matrix n n R
        preserving_snd : Finset (Equiv.Perm (Prod n o)) := Finset.filter (fun σ => ∀ ( …
        mem_preserving_snd : ∀ {σ : Equiv.Perm (Prod n o)}, Iff (Membership.mem preser …
        σ : (a : o) → Membership.mem Finset.univ a → Equiv.Perm n
        ha₁✝ : Membership.mem Finset.univ σ
        σ' : (a : o) → Membership.mem Finset.univ a → Equiv.Perm n
        ha₂✝ : Membership.mem Finset.univ σ'
        eq : Eq (Equiv.prodCongrLeft fun k => σ k ⋯) (Equiv.prodCongrLeft fun k => σ'  …
        x : o
        hx : Membership.mem Finset.univ x
        k : n
        this : ∀ (k : n) (x : o), Eq ((Equiv.prodCongrLeft fun k => σ k ⋯) { fst := k, …
        ⊢ Eq ((σ x hx) k) ((σ' x hx) k)
      -/
      simp only [prodCongrLeft_apply, Prod.mk.inj_iff] at this
      /-
        case refine_2.h.h.H
        n : Type u_2
        inst✝⁴ : DecidableEq n
        inst✝³ : Fintype n
        R : Type v
        inst✝² : CommRing R
        o : Type u_3
        inst✝¹ : Fintype o
        inst✝ : DecidableEq o
        M : o → Matrix n n R
        preserving_snd : Finset (Equiv.Perm (Prod n o)) := Finset.filter (fun σ => ∀ ( …
        mem_preserving_snd : ∀ {σ : Equiv.Perm (Prod n o)}, Iff (Membership.mem preser …
        σ : (a : o) → Membership.mem Finset.univ a → Equiv.Perm n
        ha₁✝ : Membership.mem Finset.univ σ
        σ' : (a : o) → Membership.mem Finset.univ a → Equiv.Perm n
        ha₂✝ : Membership.mem Finset.univ σ'
        eq : Eq (Equiv.prodCongrLeft fun k => σ k ⋯) (Equiv.prodCongrLeft fun k => σ'  …
        x : o
        hx : Membership.mem Finset.univ x
        k : n
        this : ∀ (k : n) (x : o), And (Eq ((σ x ⋯) k) ((σ' x ⋯) k)) True
        ⊢ Eq ((σ x hx) k) ((σ' x hx) k)
      -/
      exact (this k x).1
      /-
        🎉 no goals
      -/
      /-
        case refine_3
        n : Type u_2
        inst✝⁴ : DecidableEq n
        inst✝³ : Fintype n
        R : Type v
        inst✝² : CommRing R
        o : Type u_3
        inst✝¹ : Fintype o
        inst✝ : DecidableEq o
        M : o → Matrix n n R
        preserving_snd : Finset (Equiv.Perm (Prod n o)) := Finset.filter (fun σ => ∀ ( …
        mem_preserving_snd : ∀ {σ : Equiv.Perm (Prod n o)}, Iff (Membership.mem preser …
        ⊢ ∀ (b : Equiv (Prod n o) (Prod n o)), Membership.mem preserving_snd b → Exist …
      -/
    · intro σ hσ
      /-
        case refine_3
        n : Type u_2
        inst✝⁴ : DecidableEq n
        inst✝³ : Fintype n
        R : Type v
        inst✝² : CommRing R
        o : Type u_3
        inst✝¹ : Fintype o
        inst✝ : DecidableEq o
        M : o → Matrix n n R
        preserving_snd : Finset (Equiv.Perm (Prod n o)) := Finset.filter (fun σ => ∀ ( …
        mem_preserving_snd : ∀ {σ : Equiv.Perm (Prod n o)}, Iff (Membership.mem preser …
        σ : Equiv (Prod n o) (Prod n o)
        hσ : Membership.mem preserving_snd σ
        ⊢ Exists fun a => Exists fun ha => Eq ((fun σ x => Equiv.prodCongrLeft fun k = …
      -/
      rw [mem_preserving_snd] at hσ
      have hσ' : ∀ x, (σ⁻¹ x).snd = x.snd := by
        intro x
        conv_rhs => rw [← Perm.apply_inv_self σ x, hσ]
      have mk_apply_eq : ∀ k x, ((σ (x, k)).fst, k) = σ (x, k) := by
        intro k x
        ext
        · simp only
        · simp only [hσ]
      have mk_inv_apply_eq : ∀ k x, ((σ⁻¹ (x, k)).fst, k) = σ⁻¹ (x, k) := by
        intro k x
        conv_lhs => rw [← Perm.apply_inv_self σ (x, k)]
        ext
        · simp only [apply_inv_self]
        · simp only [hσ']
      /-
        case refine_3
        n : Type u_2
        inst✝⁴ : DecidableEq n
        inst✝³ : Fintype n
        R : Type v
        inst✝² : CommRing R
        o : Type u_3
        inst✝¹ : Fintype o
        inst✝ : DecidableEq o
        M : o → Matrix n n R
        preserving_snd : Finset (Equiv.Perm (Prod n o)) := Finset.filter (fun σ => ∀ ( …
        mem_preserving_snd : ∀ {σ : Equiv.Perm (Prod n o)}, Iff (Membership.mem preser …
        σ : Equiv (Prod n o) (Prod n o)
        hσ : ∀ (x : Prod n o), Eq (σ x).2 x.2
        hσ' : ∀ (x : Prod n o), Eq ((Inv.inv σ) x).2 x.2
        mk_apply_eq : ∀ (k : o) (x : n), Eq { fst := (σ { fst := x, snd := k }).1, snd …
        mk_inv_apply_eq : ∀ (k : o) (x : n), Eq { fst := ((Inv.inv σ) { fst := x, snd  …
        ⊢ Exists fun a => Exists fun ha => Eq ((fun σ x => Equiv.prodCongrLeft fun k = …
      -/
      refine ⟨fun k _ => ⟨fun x => (σ (x, k)).fst, fun x => (σ⁻¹ (x, k)).fst, ?_, ?_⟩, ?_, ?_⟩
        /-
          case refine_3.refine_1
          n : Type u_2
          inst✝⁴ : DecidableEq n
          inst✝³ : Fintype n
          R : Type v
          inst✝² : CommRing R
          o : Type u_3
          inst✝¹ : Fintype o
          inst✝ : DecidableEq o
          M : o → Matrix n n R
          preserving_snd : Finset (Equiv.Perm (Prod n o)) := Finset.filter (fun σ => ∀ ( …
          mem_preserving_snd : ∀ {σ : Equiv.Perm (Prod n o)}, Iff (Membership.mem preser …
          σ : Equiv (Prod n o) (Prod n o)
          hσ : ∀ (x : Prod n o), Eq (σ x).2 x.2
          hσ' : ∀ (x : Prod n o), Eq ((Inv.inv σ) x).2 x.2
          mk_apply_eq : ∀ (k : o) (x : n), Eq { fst := (σ { fst := x, snd := k }).1, snd …
          mk_inv_apply_eq : ∀ (k : o) (x : n), Eq { fst := ((Inv.inv σ) { fst := x, snd  …
          k : o
          x✝ : Membership.mem Finset.univ k
          ⊢ Function.LeftInverse (fun x => ((Inv.inv σ) { fst := x, snd := k }).1) fun x …
        -/
      · intro x
        /-
          case refine_3.refine_1
          n : Type u_2
          inst✝⁴ : DecidableEq n
          inst✝³ : Fintype n
          R : Type v
          inst✝² : CommRing R
          o : Type u_3
          inst✝¹ : Fintype o
          inst✝ : DecidableEq o
          M : o → Matrix n n R
          preserving_snd : Finset (Equiv.Perm (Prod n o)) := Finset.filter (fun σ => ∀ ( …
          mem_preserving_snd : ∀ {σ : Equiv.Perm (Prod n o)}, Iff (Membership.mem preser …
          σ : Equiv (Prod n o) (Prod n o)
          hσ : ∀ (x : Prod n o), Eq (σ x).2 x.2
          hσ' : ∀ (x : Prod n o), Eq ((Inv.inv σ) x).2 x.2
          mk_apply_eq : ∀ (k : o) (x : n), Eq { fst := (σ { fst := x, snd := k }).1, snd …
          mk_inv_apply_eq : ∀ (k : o) (x : n), Eq { fst := ((Inv.inv σ) { fst := x, snd  …
          k : o
          x✝ : Membership.mem Finset.univ k
          x : n
          ⊢ Eq ((fun x => ((Inv.inv σ) { fst := x, snd := k }).1) ((fun x => (σ { fst := …
        -/
        simp only [mk_apply_eq, inv_apply_self]
        /-
          🎉 no goals
        -/
        /-
          case refine_3.refine_2
          n : Type u_2
          inst✝⁴ : DecidableEq n
          inst✝³ : Fintype n
          R : Type v
          inst✝² : CommRing R
          o : Type u_3
          inst✝¹ : Fintype o
          inst✝ : DecidableEq o
          M : o → Matrix n n R
          preserving_snd : Finset (Equiv.Perm (Prod n o)) := Finset.filter (fun σ => ∀ ( …
          mem_preserving_snd : ∀ {σ : Equiv.Perm (Prod n o)}, Iff (Membership.mem preser …
          σ : Equiv (Prod n o) (Prod n o)
          hσ : ∀ (x : Prod n o), Eq (σ x).2 x.2
          hσ' : ∀ (x : Prod n o), Eq ((Inv.inv σ) x).2 x.2
          mk_apply_eq : ∀ (k : o) (x : n), Eq { fst := (σ { fst := x, snd := k }).1, snd …
          mk_inv_apply_eq : ∀ (k : o) (x : n), Eq { fst := ((Inv.inv σ) { fst := x, snd  …
          k : o
          x✝ : Membership.mem Finset.univ k
          ⊢ Function.RightInverse (fun x => ((Inv.inv σ) { fst := x, snd := k }).1) fun  …
        -/
      · intro x
        /-
          case refine_3.refine_2
          n : Type u_2
          inst✝⁴ : DecidableEq n
          inst✝³ : Fintype n
          R : Type v
          inst✝² : CommRing R
          o : Type u_3
          inst✝¹ : Fintype o
          inst✝ : DecidableEq o
          M : o → Matrix n n R
          preserving_snd : Finset (Equiv.Perm (Prod n o)) := Finset.filter (fun σ => ∀ ( …
          mem_preserving_snd : ∀ {σ : Equiv.Perm (Prod n o)}, Iff (Membership.mem preser …
          σ : Equiv (Prod n o) (Prod n o)
          hσ : ∀ (x : Prod n o), Eq (σ x).2 x.2
          hσ' : ∀ (x : Prod n o), Eq ((Inv.inv σ) x).2 x.2
          mk_apply_eq : ∀ (k : o) (x : n), Eq { fst := (σ { fst := x, snd := k }).1, snd …
          mk_inv_apply_eq : ∀ (k : o) (x : n), Eq { fst := ((Inv.inv σ) { fst := x, snd  …
          k : o
          x✝ : Membership.mem Finset.univ k
          x : n
          ⊢ Eq ((fun x => (σ { fst := x, snd := k }).1) ((fun x => ((Inv.inv σ) { fst := …
        -/
        simp only [mk_inv_apply_eq, apply_inv_self]
        /-
          🎉 no goals
        -/
        /-
          case refine_3.refine_3
          n : Type u_2
          inst✝⁴ : DecidableEq n
          inst✝³ : Fintype n
          R : Type v
          inst✝² : CommRing R
          o : Type u_3
          inst✝¹ : Fintype o
          inst✝ : DecidableEq o
          M : o → Matrix n n R
          preserving_snd : Finset (Equiv.Perm (Prod n o)) := Finset.filter (fun σ => ∀ ( …
          mem_preserving_snd : ∀ {σ : Equiv.Perm (Prod n o)}, Iff (Membership.mem preser …
          σ : Equiv (Prod n o) (Prod n o)
          hσ : ∀ (x : Prod n o), Eq (σ x).2 x.2
          hσ' : ∀ (x : Prod n o), Eq ((Inv.inv σ) x).2 x.2
          mk_apply_eq : ∀ (k : o) (x : n), Eq { fst := (σ { fst := x, snd := k }).1, snd …
          mk_inv_apply_eq : ∀ (k : o) (x : n), Eq { fst := ((Inv.inv σ) { fst := x, snd  …
          ⊢ Membership.mem Finset.univ fun k x => { toFun := fun x => (σ { fst := x, snd …
        -/
      · apply Finset.mem_univ
        /-
          🎉 no goals
        -/
        /-
          case refine_3.refine_4
          n : Type u_2
          inst✝⁴ : DecidableEq n
          inst✝³ : Fintype n
          R : Type v
          inst✝² : CommRing R
          o : Type u_3
          inst✝¹ : Fintype o
          inst✝ : DecidableEq o
          M : o → Matrix n n R
          preserving_snd : Finset (Equiv.Perm (Prod n o)) := Finset.filter (fun σ => ∀ ( …
          mem_preserving_snd : ∀ {σ : Equiv.Perm (Prod n o)}, Iff (Membership.mem preser …
          σ : Equiv (Prod n o) (Prod n o)
          hσ : ∀ (x : Prod n o), Eq (σ x).2 x.2
          hσ' : ∀ (x : Prod n o), Eq ((Inv.inv σ) x).2 x.2
          mk_apply_eq : ∀ (k : o) (x : n), Eq { fst := (σ { fst := x, snd := k }).1, snd …
          mk_inv_apply_eq : ∀ (k : o) (x : n), Eq { fst := ((Inv.inv σ) { fst := x, snd  …
          ⊢ Eq ((fun σ x => Equiv.prodCongrLeft fun k => σ k ⋯) (fun k x => { toFun := f …
        -/
      · ext ⟨k, x⟩
          /-
            case refine_3.refine_4.H.mk.fst
            n : Type u_2
            inst✝⁴ : DecidableEq n
            inst✝³ : Fintype n
            R : Type v
            inst✝² : CommRing R
            o : Type u_3
            inst✝¹ : Fintype o
            inst✝ : DecidableEq o
            M : o → Matrix n n R
            preserving_snd : Finset (Equiv.Perm (Prod n o)) := Finset.filter (fun σ => ∀ ( …
            mem_preserving_snd : ∀ {σ : Equiv.Perm (Prod n o)}, Iff (Membership.mem preser …
            σ : Equiv (Prod n o) (Prod n o)
            hσ : ∀ (x : Prod n o), Eq (σ x).2 x.2
            hσ' : ∀ (x : Prod n o), Eq ((Inv.inv σ) x).2 x.2
            mk_apply_eq : ∀ (k : o) (x : n), Eq { fst := (σ { fst := x, snd := k }).1, snd …
            mk_inv_apply_eq : ∀ (k : o) (x : n), Eq { fst := ((Inv.inv σ) { fst := x, snd  …
            k : n
            x : o
            ⊢ Eq (((fun σ x => Equiv.prodCongrLeft fun k => σ k ⋯) (fun k x => { toFun :=  …
          -/
        · simp only [coe_fn_mk, prodCongrLeft_apply]
          /-
            🎉 no goals
          -/
          /-
            case refine_3.refine_4.H.mk.snd
            n : Type u_2
            inst✝⁴ : DecidableEq n
            inst✝³ : Fintype n
            R : Type v
            inst✝² : CommRing R
            o : Type u_3
            inst✝¹ : Fintype o
            inst✝ : DecidableEq o
            M : o → Matrix n n R
            preserving_snd : Finset (Equiv.Perm (Prod n o)) := Finset.filter (fun σ => ∀ ( …
            mem_preserving_snd : ∀ {σ : Equiv.Perm (Prod n o)}, Iff (Membership.mem preser …
            σ : Equiv (Prod n o) (Prod n o)
            hσ : ∀ (x : Prod n o), Eq (σ x).2 x.2
            hσ' : ∀ (x : Prod n o), Eq ((Inv.inv σ) x).2 x.2
            mk_apply_eq : ∀ (k : o) (x : n), Eq { fst := (σ { fst := x, snd := k }).1, snd …
            mk_inv_apply_eq : ∀ (k : o) (x : n), Eq { fst := ((Inv.inv σ) { fst := x, snd  …
            k : n
            x : o
            ⊢ Eq (((fun σ x => Equiv.prodCongrLeft fun k => σ k ⋯) (fun k x => { toFun :=  …
          -/
        · simp only [prodCongrLeft_apply, hσ]
          /-
            🎉 no goals
          -/
      /-
        case refine_4
        n : Type u_2
        inst✝⁴ : DecidableEq n
        inst✝³ : Fintype n
        R : Type v
        inst✝² : CommRing R
        o : Type u_3
        inst✝¹ : Fintype o
        inst✝ : DecidableEq o
        M : o → Matrix n n R
        preserving_snd : Finset (Equiv.Perm (Prod n o)) := Finset.filter (fun σ => ∀ ( …
        mem_preserving_snd : ∀ {σ : Equiv.Perm (Prod n o)}, Iff (Membership.mem preser …
        ⊢ ∀ (a : (a : o) → Membership.mem Finset.univ a → Equiv.Perm n) (ha : Membersh …
      -/
    · intro σ _
      /-
        case refine_4
        n : Type u_2
        inst✝⁴ : DecidableEq n
        inst✝³ : Fintype n
        R : Type v
        inst✝² : CommRing R
        o : Type u_3
        inst✝¹ : Fintype o
        inst✝ : DecidableEq o
        M : o → Matrix n n R
        preserving_snd : Finset (Equiv.Perm (Prod n o)) := Finset.filter (fun σ => ∀ ( …
        mem_preserving_snd : ∀ {σ : Equiv.Perm (Prod n o)}, Iff (Membership.mem preser …
        σ : (a : o) → Membership.mem Finset.univ a → Equiv.Perm n
        ha✝ : Membership.mem Finset.univ σ
        ⊢ Eq (Finset.univ.prod fun x => HMul.hMul (↑↑(Equiv.Perm.sign (σ x ⋯))) (Finse …
      -/
      rw [Finset.prod_mul_distrib, ← Finset.univ_product_univ, Finset.prod_product_right]
      simp only [sign_prodCongrLeft, Units.coe_prod, Int.cast_prod, blockDiagonal_apply_eq,
        prodCongrLeft_apply]
    /-
      n : Type u_2
      inst✝⁴ : DecidableEq n
      inst✝³ : Fintype n
      R : Type v
      inst✝² : CommRing R
      o : Type u_3
      inst✝¹ : Fintype o
      inst✝ : DecidableEq o
      M : o → Matrix n n R
      preserving_snd : Finset (Equiv.Perm (Prod n o)) := Finset.filter (fun σ => ∀ ( …
      mem_preserving_snd : ∀ {σ : Equiv.Perm (Prod n o)}, Iff (Membership.mem preser …
      ⊢ ∀ (x : Equiv.Perm (Prod n o)), Membership.mem Finset.univ x → Not (Membershi …
    -/
  · intro σ _ hσ
    /-
      n : Type u_2
      inst✝⁴ : DecidableEq n
      inst✝³ : Fintype n
      R : Type v
      inst✝² : CommRing R
      o : Type u_3
      inst✝¹ : Fintype o
      inst✝ : DecidableEq o
      M : o → Matrix n n R
      preserving_snd : Finset (Equiv.Perm (Prod n o)) := Finset.filter (fun σ => ∀ ( …
      mem_preserving_snd : ∀ {σ : Equiv.Perm (Prod n o)}, Iff (Membership.mem preser …
      σ : Equiv.Perm (Prod n o)
      a✝ : Membership.mem Finset.univ σ
      hσ : Not (Membership.mem preserving_snd σ)
      ⊢ Eq (HMul.hMul (↑↑(Equiv.Perm.sign σ)) (Finset.univ.prod fun i => Matrix.bloc …
    -/
    rw [mem_preserving_snd] at hσ
    /-
      n : Type u_2
      inst✝⁴ : DecidableEq n
      inst✝³ : Fintype n
      R : Type v
      inst✝² : CommRing R
      o : Type u_3
      inst✝¹ : Fintype o
      inst✝ : DecidableEq o
      M : o → Matrix n n R
      preserving_snd : Finset (Equiv.Perm (Prod n o)) := Finset.filter (fun σ => ∀ ( …
      mem_preserving_snd : ∀ {σ : Equiv.Perm (Prod n o)}, Iff (Membership.mem preser …
      σ : Equiv.Perm (Prod n o)
      a✝ : Membership.mem Finset.univ σ
      hσ : Not (∀ (x : Prod n o), Eq (σ x).2 x.2)
      ⊢ Eq (HMul.hMul (↑↑(Equiv.Perm.sign σ)) (Finset.univ.prod fun i => Matrix.bloc …
    -/
    obtain ⟨⟨k, x⟩, hkx⟩ := not_forall.mp hσ
    /-
      case intro.mk
      n : Type u_2
      inst✝⁴ : DecidableEq n
      inst✝³ : Fintype n
      R : Type v
      inst✝² : CommRing R
      o : Type u_3
      inst✝¹ : Fintype o
      inst✝ : DecidableEq o
      M : o → Matrix n n R
      preserving_snd : Finset (Equiv.Perm (Prod n o)) := Finset.filter (fun σ => ∀ ( …
      mem_preserving_snd : ∀ {σ : Equiv.Perm (Prod n o)}, Iff (Membership.mem preser …
      σ : Equiv.Perm (Prod n o)
      a✝ : Membership.mem Finset.univ σ
      hσ : Not (∀ (x : Prod n o), Eq (σ x).2 x.2)
      k : n
      x : o
      hkx : Not (Eq (σ { fst := k, snd := x }).2 { fst := k, snd := x }.2)
      ⊢ Eq (HMul.hMul (↑↑(Equiv.Perm.sign σ)) (Finset.univ.prod fun i => Matrix.bloc …
    -/
    rw [Finset.prod_eq_zero (Finset.mem_univ (k, x)), mul_zero]
    /-
      case intro.mk
      n : Type u_2
      inst✝⁴ : DecidableEq n
      inst✝³ : Fintype n
      R : Type v
      inst✝² : CommRing R
      o : Type u_3
      inst✝¹ : Fintype o
      inst✝ : DecidableEq o
      M : o → Matrix n n R
      preserving_snd : Finset (Equiv.Perm (Prod n o)) := Finset.filter (fun σ => ∀ ( …
      mem_preserving_snd : ∀ {σ : Equiv.Perm (Prod n o)}, Iff (Membership.mem preser …
      σ : Equiv.Perm (Prod n o)
      a✝ : Membership.mem Finset.univ σ
      hσ : Not (∀ (x : Prod n o), Eq (σ x).2 x.2)
      k : n
      x : o
      hkx : Not (Eq (σ { fst := k, snd := x }).2 { fst := k, snd := x }.2)
      ⊢ Eq (Matrix.blockDiagonal M (σ { fst := k, snd := x }) { fst := k, snd := x } …
    -/
    rw [blockDiagonal_apply_ne]
    /-
      case intro.mk.h
      n : Type u_2
      inst✝⁴ : DecidableEq n
      inst✝³ : Fintype n
      R : Type v
      inst✝² : CommRing R
      o : Type u_3
      inst✝¹ : Fintype o
      inst✝ : DecidableEq o
      M : o → Matrix n n R
      preserving_snd : Finset (Equiv.Perm (Prod n o)) := Finset.filter (fun σ => ∀ ( …
      mem_preserving_snd : ∀ {σ : Equiv.Perm (Prod n o)}, Iff (Membership.mem preser …
      σ : Equiv.Perm (Prod n o)
      a✝ : Membership.mem Finset.univ σ
      hσ : Not (∀ (x : Prod n o), Eq (σ x).2 x.2)
      k : n
      x : o
      hkx : Not (Eq (σ { fst := k, snd := x }).2 { fst := k, snd := x }.2)
      ⊢ Ne (σ { fst := k, snd := x }).2 x
    -/
    exact hkx
    /-
      🎉 no goals
    -/


/-- The determinant of a 2×2 block matrix with the lower-left block equal to zero is the product of
the determinants of the diagonal blocks. For the generalization to any number of blocks, see
`Matrix.det_of_upperTriangular`. -/
@[simp]
theorem det_fromBlocks_zero₂₁ (A : Matrix m m R) (B : Matrix m n R) (D : Matrix n n R) :
    (Matrix.fromBlocks A B 0 D).det = A.det * D.det := by
  classical
    simp_rw [det_apply']
    convert Eq.symm <|
      sum_subset (β := R) (subset_univ ((sumCongrHom m n).range : Set (Perm (m ⊕ n))).toFinset) ?_
    · simp_rw [sum_mul_sum, ← sum_product', univ_product_univ]
      refine sum_nbij (fun σ ↦ σ.fst.sumCongr σ.snd) ?_ ?_ ?_ ?_
      · intro σ₁₂ _
        simp only
        erw [Set.mem_toFinset, MonoidHom.mem_range]
        use σ₁₂
        simp only [sumCongrHom_apply]
      · intro σ₁ _ σ₂ _
        dsimp only
        intro h
        have h2 : ∀ x, Perm.sumCongr σ₁.fst σ₁.snd x = Perm.sumCongr σ₂.fst σ₂.snd x :=
          DFunLike.congr_fun h
        simp only [Sum.map_inr, Sum.map_inl, Perm.sumCongr_apply, Sum.forall, Sum.inl.injEq,
          Sum.inr.injEq] at h2
        ext x
        · exact h2.left x
        · exact h2.right x
      · intro σ hσ
        erw [Set.mem_toFinset, MonoidHom.mem_range] at hσ
        obtain ⟨σ₁₂, hσ₁₂⟩ := hσ
        use σ₁₂
        rw [← hσ₁₂]
        simp
      · simp only [forall_prop_of_true, Prod.forall, mem_univ]
        intro σ₁ σ₂
        rw [Fintype.prod_sum_type]
        simp_rw [Equiv.sumCongr_apply, Sum.map_inr, Sum.map_inl, fromBlocks_apply₁₁,
          fromBlocks_apply₂₂]
        rw [mul_mul_mul_comm]
        congr
        rw [sign_sumCongr, Units.val_mul, Int.cast_mul]
    · rintro σ - hσn
      have h1 : ¬∀ x, ∃ y, Sum.inl y = σ (Sum.inl x) := by
        rw [Set.mem_toFinset] at hσn
        -- Porting note: golfed
        simpa only [Set.MapsTo, Set.mem_range, forall_exists_index, forall_apply_eq_imp_iff] using
          mt mem_sumCongrHom_range_of_perm_mapsTo_inl hσn
      obtain ⟨a, ha⟩ := not_forall.mp h1
      cases' hx : σ (Sum.inl a) with a2 b
      · have hn := (not_exists.mp ha) a2
        exact absurd hx.symm hn
      · rw [Finset.prod_eq_zero (Finset.mem_univ (Sum.inl a)), mul_zero]
        rw [hx, fromBlocks_apply₂₁, zero_apply]


/-- The determinant of a 2×2 block matrix with the upper-right block equal to zero is the product of
the determinants of the diagonal blocks. For the generalization to any number of blocks, see
`Matrix.det_of_lowerTriangular`. -/
@[simp]
theorem det_fromBlocks_zero₁₂ (A : Matrix m m R) (C : Matrix n m R) (D : Matrix n n R) :
    (Matrix.fromBlocks A 0 C D).det = A.det * D.det := by
  rw [← det_transpose, fromBlocks_transpose, transpose_zero, det_fromBlocks_zero₂₁, det_transpose,
    det_transpose]


/-- Laplacian expansion of the determinant of an `n+1 × n+1` matrix along column 0. -/
theorem det_succ_column_zero {n : ℕ} (A : Matrix (Fin n.succ) (Fin n.succ) R) :
    det A = ∑ i : Fin n.succ, (-1) ^ (i : ℕ) * A i 0 * det (A.submatrix i.succAbove Fin.succ) := by
  /-
    R : Type v
    inst✝ : CommRing R
    n : Nat
    A : Matrix (Fin n.succ) (Fin n.succ) R
    ⊢ Eq A.det (Finset.univ.sum fun i => HMul.hMul (HMul.hMul (HPow.hPow (-1) ↑i)  …
  -/
  rw [Matrix.det_apply, Finset.univ_perm_fin_succ, ← Finset.univ_product_univ]
  /-
    R : Type v
    inst✝ : CommRing R
    n : Nat
    A : Matrix (Fin n.succ) (Fin n.succ) R
    ⊢ Eq ((Finset.map Equiv.Perm.decomposeFin.symm.toEmbedding (SProd.sprod Finset …
  -/
  simp only [Finset.sum_map, Equiv.toEmbedding_apply, Finset.sum_product, Matrix.submatrix]
  /-
    R : Type v
    inst✝ : CommRing R
    n : Nat
    A : Matrix (Fin n.succ) (Fin n.succ) R
    ⊢ Eq (Finset.univ.sum fun x => Finset.univ.sum fun y => HSMul.hSMul (Equiv.Per …
  -/
  refine Finset.sum_congr rfl fun i _ => Fin.cases ?_ (fun i => ?_) i
  · simp only [Fin.prod_univ_succ, Matrix.det_apply, Finset.mul_sum,
      Equiv.Perm.decomposeFin_symm_apply_zero, Fin.val_zero, one_mul,
      Equiv.Perm.decomposeFin.symm_sign, Equiv.swap_self, if_true, id, eq_self_iff_true,
      Equiv.Perm.decomposeFin_symm_apply_succ, Fin.succAbove_zero, Equiv.coe_refl, pow_zero,
      mul_smul_comm, of_apply]
  -- `univ_perm_fin_succ` gives a different embedding of `Perm (Fin n)` into
  -- `Perm (Fin n.succ)` than the determinant of the submatrix we want,
  -- permute `A` so that we get the correct one.
  /-
    case refine_2
    R : Type v
    inst✝ : CommRing R
    n : Nat
    A : Matrix (Fin n.succ) (Fin n.succ) R
    i✝ : Fin n.succ
    x✝ : Membership.mem Finset.univ i✝
    i : Fin n
    ⊢ Eq (Finset.univ.sum fun y => HSMul.hSMul (Equiv.Perm.sign (Equiv.Perm.decomp …
  -/
  have : (-1 : R) ^ (i : ℕ) = (Perm.sign i.cycleRange) := by simp [Fin.sign_cycleRange]
  rw [Fin.val_succ, pow_succ', this, mul_assoc, mul_assoc, mul_left_comm (ε _),
    ← det_permute, Matrix.det_apply, Finset.mul_sum, Finset.mul_sum]
  -- now we just need to move the corresponding parts to the same place
  /-
    case refine_2
    R : Type v
    inst✝ : CommRing R
    n : Nat
    A : Matrix (Fin n.succ) (Fin n.succ) R
    i✝ : Fin n.succ
    x✝ : Membership.mem Finset.univ i✝
    i : Fin n
    this : Eq (HPow.hPow (-1) ↑i) ↑↑(Equiv.Perm.sign i.cycleRange)
    ⊢ Eq (Finset.univ.sum fun y => HSMul.hSMul (Equiv.Perm.sign (Equiv.Perm.decomp …
  -/
  refine Finset.sum_congr rfl fun σ _ => ?_
  /-
    case refine_2
    R : Type v
    inst✝ : CommRing R
    n : Nat
    A : Matrix (Fin n.succ) (Fin n.succ) R
    i✝ : Fin n.succ
    x✝¹ : Membership.mem Finset.univ i✝
    i : Fin n
    this : Eq (HPow.hPow (-1) ↑i) ↑↑(Equiv.Perm.sign i.cycleRange)
    σ : Equiv.Perm (Fin n)
    x✝ : Membership.mem Finset.univ σ
    ⊢ Eq (HSMul.hSMul (Equiv.Perm.sign (Equiv.Perm.decomposeFin.symm { fst := i.su …
  -/
  rw [Equiv.Perm.decomposeFin.symm_sign, if_neg (Fin.succ_ne_zero i)]
  calc
    ((-1 * Perm.sign σ : ℤ) • ∏ i', A (Perm.decomposeFin.symm (Fin.succ i, σ) i') i') =
        (-1 * Perm.sign σ : ℤ) • (A (Fin.succ i) 0 *
          ∏ i', A ((Fin.succ i).succAbove (Fin.cycleRange i (σ i'))) i'.succ) := by
      simp only [Fin.prod_univ_succ, Fin.succAbove_cycleRange,
        Equiv.Perm.decomposeFin_symm_apply_zero, Equiv.Perm.decomposeFin_symm_apply_succ]
    _ = -1 * (A (Fin.succ i) 0 * (Perm.sign σ : ℤ) •
        ∏ i', A ((Fin.succ i).succAbove (Fin.cycleRange i (σ i'))) i'.succ) := by
      simp [mul_assoc, mul_comm, _root_.neg_mul, one_mul, zsmul_eq_mul, neg_inj, neg_smul,
        Fin.succAbove_cycleRange, mul_left_comm]


/-- Laplacian expansion of the determinant of an `n+1 × n+1` matrix along row 0. -/
theorem det_succ_row_zero {n : ℕ} (A : Matrix (Fin n.succ) (Fin n.succ) R) :
    det A = ∑ j : Fin n.succ, (-1) ^ (j : ℕ) * A 0 j * det (A.submatrix Fin.succ j.succAbove) := by
  /-
    R : Type v
    inst✝ : CommRing R
    n : Nat
    A : Matrix (Fin n.succ) (Fin n.succ) R
    ⊢ Eq A.det (Finset.univ.sum fun j => HMul.hMul (HMul.hMul (HPow.hPow (-1) ↑j)  …
  -/
  rw [← det_transpose A, det_succ_column_zero]
  /-
    R : Type v
    inst✝ : CommRing R
    n : Nat
    A : Matrix (Fin n.succ) (Fin n.succ) R
    ⊢ Eq (Finset.univ.sum fun i => HMul.hMul (HMul.hMul (HPow.hPow (-1) ↑i) (A.tra …
  -/
  refine Finset.sum_congr rfl fun i _ => ?_
  /-
    R : Type v
    inst✝ : CommRing R
    n : Nat
    A : Matrix (Fin n.succ) (Fin n.succ) R
    i : Fin n.succ
    x✝ : Membership.mem Finset.univ i
    ⊢ Eq (HMul.hMul (HMul.hMul (HPow.hPow (-1) ↑i) (A.transpose i 0)) (A.transpose …
  -/
  rw [← det_transpose]
  /-
    R : Type v
    inst✝ : CommRing R
    n : Nat
    A : Matrix (Fin n.succ) (Fin n.succ) R
    i : Fin n.succ
    x✝ : Membership.mem Finset.univ i
    ⊢ Eq (HMul.hMul (HMul.hMul (HPow.hPow (-1) ↑i) (A.transpose i 0)) (A.transpose …
  -/
  simp only [transpose_apply, transpose_submatrix, transpose_transpose]
  /-
    🎉 no goals
  -/


/-- Laplacian expansion of the determinant of an `n+1 × n+1` matrix along row `i`. -/
theorem det_succ_row {n : ℕ} (A : Matrix (Fin n.succ) (Fin n.succ) R) (i : Fin n.succ) :
    det A =
      ∑ j : Fin n.succ, (-1) ^ (i + j : ℕ) * A i j * det (A.submatrix i.succAbove j.succAbove) := by
  /-
    R : Type v
    inst✝ : CommRing R
    n : Nat
    A : Matrix (Fin n.succ) (Fin n.succ) R
    i : Fin n.succ
    ⊢ Eq A.det (Finset.univ.sum fun j => HMul.hMul (HMul.hMul (HPow.hPow (-1) (HAd …
  -/
  simp_rw [pow_add, mul_assoc, ← mul_sum]
  have : det A = (-1 : R) ^ (i : ℕ) * (Perm.sign i.cycleRange⁻¹) * det A := by
    calc
      det A = ↑((-1 : ℤˣ) ^ (i : ℕ) * (-1 : ℤˣ) ^ (i : ℕ) : ℤˣ) * det A := by simp
      _ = (-1 : R) ^ (i : ℕ) * (Perm.sign i.cycleRange⁻¹) * det A := by simp [-Int.units_mul_self]
  /-
    R : Type v
    inst✝ : CommRing R
    n : Nat
    A : Matrix (Fin n.succ) (Fin n.succ) R
    i : Fin n.succ
    this : Eq A.det (HMul.hMul (HMul.hMul (HPow.hPow (-1) ↑i) ↑↑(Equiv.Perm.sign ( …
    ⊢ Eq A.det (HMul.hMul (HPow.hPow (-1) ↑i) (Finset.univ.sum fun i_1 => HMul.hMu …
  -/
  rw [this, mul_assoc]
  /-
    R : Type v
    inst✝ : CommRing R
    n : Nat
    A : Matrix (Fin n.succ) (Fin n.succ) R
    i : Fin n.succ
    this : Eq A.det (HMul.hMul (HMul.hMul (HPow.hPow (-1) ↑i) ↑↑(Equiv.Perm.sign ( …
    ⊢ Eq (HMul.hMul (HPow.hPow (-1) ↑i) (HMul.hMul (↑↑(Equiv.Perm.sign (Inv.inv i. …
  -/
  congr
  /-
    case e_a
    R : Type v
    inst✝ : CommRing R
    n : Nat
    A : Matrix (Fin n.succ) (Fin n.succ) R
    i : Fin n.succ
    this : Eq A.det (HMul.hMul (HMul.hMul (HPow.hPow (-1) ↑i) ↑↑(Equiv.Perm.sign ( …
    ⊢ Eq (HMul.hMul (↑↑(Equiv.Perm.sign (Inv.inv i.cycleRange))) A.det) (Finset.un …
  -/
  rw [← det_permute, det_succ_row_zero]
  /-
    case e_a
    R : Type v
    inst✝ : CommRing R
    n : Nat
    A : Matrix (Fin n.succ) (Fin n.succ) R
    i : Fin n.succ
    this : Eq A.det (HMul.hMul (HMul.hMul (HPow.hPow (-1) ↑i) ↑↑(Equiv.Perm.sign ( …
    ⊢ Eq (Finset.univ.sum fun j => HMul.hMul (HMul.hMul (HPow.hPow (-1) ↑j) (A.sub …
  -/
  refine Finset.sum_congr rfl fun j _ => ?_
  /-
    case e_a
    R : Type v
    inst✝ : CommRing R
    n : Nat
    A : Matrix (Fin n.succ) (Fin n.succ) R
    i : Fin n.succ
    this : Eq A.det (HMul.hMul (HMul.hMul (HPow.hPow (-1) ↑i) ↑↑(Equiv.Perm.sign ( …
    j : Fin n.succ
    x✝ : Membership.mem Finset.univ j
    ⊢ Eq (HMul.hMul (HMul.hMul (HPow.hPow (-1) ↑j) (A.submatrix (⇑(Inv.inv i.cycle …
  -/
  rw [mul_assoc, Matrix.submatrix_apply, submatrix_submatrix, id_comp, Function.comp_def, id]
  /-
    case e_a
    R : Type v
    inst✝ : CommRing R
    n : Nat
    A : Matrix (Fin n.succ) (Fin n.succ) R
    i : Fin n.succ
    this : Eq A.det (HMul.hMul (HMul.hMul (HPow.hPow (-1) ↑i) ↑↑(Equiv.Perm.sign ( …
    j : Fin n.succ
    x✝ : Membership.mem Finset.univ j
    ⊢ Eq (HMul.hMul (HPow.hPow (-1) ↑j) (HMul.hMul (A ((Inv.inv i.cycleRange) 0) j …
  -/
  congr
    /-
      case e_a.e_a.e_a.e_a
      R : Type v
      inst✝ : CommRing R
      n : Nat
      A : Matrix (Fin n.succ) (Fin n.succ) R
      i : Fin n.succ
      this : Eq A.det (HMul.hMul (HMul.hMul (HPow.hPow (-1) ↑i) ↑↑(Equiv.Perm.sign ( …
      j : Fin n.succ
      x✝ : Membership.mem Finset.univ j
      ⊢ Eq ((Inv.inv i.cycleRange) 0) i
    -/
  · rw [Equiv.Perm.inv_def, Fin.cycleRange_symm_zero]
    /-
      🎉 no goals
    -/
    /-
      case e_a.e_a.e_a.e_M
      R : Type v
      inst✝ : CommRing R
      n : Nat
      A : Matrix (Fin n.succ) (Fin n.succ) R
      i : Fin n.succ
      this : Eq A.det (HMul.hMul (HMul.hMul (HPow.hPow (-1) ↑i) ↑↑(Equiv.Perm.sign ( …
      j : Fin n.succ
      x✝ : Membership.mem Finset.univ j
      ⊢ Eq (A.submatrix (fun x => (Inv.inv i.cycleRange) x.succ) j.succAbove) (A.sub …
    -/
  · ext i' j'
    rw [Equiv.Perm.inv_def, Matrix.submatrix_apply, Matrix.submatrix_apply,
      Fin.cycleRange_symm_succ]


/-- Laplacian expansion of the determinant of an `n+1 × n+1` matrix along column `j`. -/
theorem det_succ_column {n : ℕ} (A : Matrix (Fin n.succ) (Fin n.succ) R) (j : Fin n.succ) :
    det A =
      ∑ i : Fin n.succ, (-1) ^ (i + j : ℕ) * A i j * det (A.submatrix i.succAbove j.succAbove) := by
  /-
    R : Type v
    inst✝ : CommRing R
    n : Nat
    A : Matrix (Fin n.succ) (Fin n.succ) R
    j : Fin n.succ
    ⊢ Eq A.det (Finset.univ.sum fun i => HMul.hMul (HMul.hMul (HPow.hPow (-1) (HAd …
  -/
  rw [← det_transpose, det_succ_row _ j]
  /-
    R : Type v
    inst✝ : CommRing R
    n : Nat
    A : Matrix (Fin n.succ) (Fin n.succ) R
    j : Fin n.succ
    ⊢ Eq (Finset.univ.sum fun j_1 => HMul.hMul (HMul.hMul (HPow.hPow (-1) (HAdd.hA …
  -/
  refine Finset.sum_congr rfl fun i _ => ?_
  /-
    R : Type v
    inst✝ : CommRing R
    n : Nat
    A : Matrix (Fin n.succ) (Fin n.succ) R
    j i : Fin n.succ
    x✝ : Membership.mem Finset.univ i
    ⊢ Eq (HMul.hMul (HMul.hMul (HPow.hPow (-1) (HAdd.hAdd ↑j ↑i)) (A.transpose j i …
  -/
  rw [add_comm, ← det_transpose, transpose_apply, transpose_submatrix, transpose_transpose]
  /-
    🎉 no goals
  -/


/-- Determinant of 0x0 matrix -/
@[simp]
theorem det_fin_zero {A : Matrix (Fin 0) (Fin 0) R} : det A = 1 :=
  det_isEmpty


/-- Determinant of 1x1 matrix -/
theorem det_fin_one (A : Matrix (Fin 1) (Fin 1) R) : det A = A 0 0 :=
  det_unique A


theorem det_fin_one_of (a : R) : det !![a] = a :=
  det_fin_one _


/-- Determinant of 2x2 matrix -/
theorem det_fin_two (A : Matrix (Fin 2) (Fin 2) R) : det A = A 0 0 * A 1 1 - A 0 1 * A 1 0 := by
  simp only [det_succ_row_zero, det_unique, Fin.default_eq_zero, submatrix_apply,
    Fin.succ_zero_eq_one, Fin.sum_univ_succ, Fin.val_zero, Fin.zero_succAbove, univ_unique,
    Fin.val_succ, Fin.val_eq_zero, Fin.succ_succAbove_zero, sum_singleton]
  /-
    R : Type v
    inst✝ : CommRing R
    A : Matrix (Fin 2) (Fin 2) R
    ⊢ Eq (HAdd.hAdd (HMul.hMul (HMul.hMul (HPow.hPow (-1) 0) (A 0 0)) (A 1 1)) (HM …
  -/
  ring
  /-
    🎉 no goals
  -/


@[simp]
theorem det_fin_two_of (a b c d : R) : Matrix.det !![a, b; c, d] = a * d - b * c :=
  det_fin_two _


/-- Determinant of 3x3 matrix -/
theorem det_fin_three (A : Matrix (Fin 3) (Fin 3) R) :
    det A =
      A 0 0 * A 1 1 * A 2 2 - A 0 0 * A 1 2 * A 2 1
      - A 0 1 * A 1 0 * A 2 2 + A 0 1 * A 1 2 * A 2 0
      + A 0 2 * A 1 0 * A 2 1 - A 0 2 * A 1 1 * A 2 0 := by
  simp only [det_succ_row_zero, submatrix_apply, Fin.succ_zero_eq_one, submatrix_submatrix,
    det_unique, Fin.default_eq_zero, Function.comp_apply, Fin.succ_one_eq_two, Fin.sum_univ_succ,
    Fin.val_zero, Fin.zero_succAbove, univ_unique, Fin.val_succ, Fin.val_eq_zero,
    Fin.succ_succAbove_zero, sum_singleton, Fin.succ_succAbove_one]
  /-
    R : Type v
    inst✝ : CommRing R
    A : Matrix (Fin 3) (Fin 3) R
    ⊢ Eq (HAdd.hAdd (HMul.hMul (HMul.hMul (HPow.hPow (-1) 0) (A 0 0)) (HAdd.hAdd ( …
  -/
  ring
  /-
    🎉 no goals
  -/


