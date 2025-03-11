/-- A "linear recurrence relation" over a commutative semiring is given by its
  order `n` and `n` coefficients. -/
structure LinearRecurrence (α : Type*) [CommSemiring α] where
  order : ℕ
  coeffs : Fin order → α


instance (α : Type*) [CommSemiring α] : Inhabited (LinearRecurrence α) :=
  ⟨⟨0, default⟩⟩


/-- We say that a sequence `u` is solution of `LinearRecurrence order coeffs` when we have
  `u (n + order) = ∑ i : Fin order, coeffs i * u (n + i)` for any `n`. -/
def IsSolution (u : ℕ → α) :=
  ∀ n, u (n + E.order) = ∑ i, E.coeffs i * u (n + i)


/-- A solution of a `LinearRecurrence` which satisfies certain initial conditions.
  We will prove this is the only such solution. -/
def mkSol (init : Fin E.order → α) : ℕ → α
  | n =>
    if h : n < E.order then init ⟨n, h⟩
    else
      ∑ k : Fin E.order,
                                           /-
                                             α : Type u_1
                                             inst✝ : CommSemiring α
                                             E : LinearRecurrence α
                                             init : Fin E.order → α
                                             x✝ : Nat
                                             n : Nat := x✝
                                             h : Not (LT.lt n E.order)
                                             k : Fin E.order
                                             ⊢ LT.lt (HAdd.hAdd (HSub.hSub n E.order) ↑k) n
                                           -/
        have _ : n - E.order + k < n := by omega
                                           /-
                                             🎉 no goals
                                           -/
        E.coeffs k * mkSol init (n - E.order + k)


/-- `E.mkSol` indeed gives solutions to `E`. -/
theorem is_sol_mkSol (init : Fin E.order → α) : E.IsSolution (E.mkSol init) := by
  /-
    α : Type u_1
    inst✝ : CommSemiring α
    E : LinearRecurrence α
    init : Fin E.order → α
    ⊢ E.IsSolution (E.mkSol init)
  -/
  intro n
  /-
    α : Type u_1
    inst✝ : CommSemiring α
    E : LinearRecurrence α
    init : Fin E.order → α
    n : Nat
    ⊢ Eq (E.mkSol init (HAdd.hAdd n E.order)) (Finset.univ.sum fun i => HMul.hMul  …
  -/
  rw [mkSol]
  /-
    α : Type u_1
    inst✝ : CommSemiring α
    E : LinearRecurrence α
    init : Fin E.order → α
    n : Nat
    ⊢ Eq (dite (LT.lt (HAdd.hAdd n E.order) E.order) (fun h => init ⟨HAdd.hAdd n E …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- `E.mkSol init`'s first `E.order` terms are `init`. -/
theorem mkSol_eq_init (init : Fin E.order → α) : ∀ n : Fin E.order, E.mkSol init n = init n := by
  /-
    α : Type u_1
    inst✝ : CommSemiring α
    E : LinearRecurrence α
    init : Fin E.order → α
    ⊢ ∀ (n : Fin E.order), Eq (E.mkSol init ↑n) (init n)
  -/
  intro n
  /-
    α : Type u_1
    inst✝ : CommSemiring α
    E : LinearRecurrence α
    init : Fin E.order → α
    n : Fin E.order
    ⊢ Eq (E.mkSol init ↑n) (init n)
  -/
  rw [mkSol]
  /-
    α : Type u_1
    inst✝ : CommSemiring α
    E : LinearRecurrence α
    init : Fin E.order → α
    n : Fin E.order
    ⊢ Eq (dite (LT.lt (↑n) E.order) (fun h => init ⟨↑n, h⟩) fun h => Finset.univ.s …
  -/
  simp only [n.is_lt, dif_pos, Fin.mk_val, Fin.eta]
  /-
    🎉 no goals
  -/


/-- If `u` is a solution to `E` and `init` designates its first `E.order` values,
  then `∀ n, u n = E.mkSol init n`. -/
theorem eq_mk_of_is_sol_of_eq_init {u : ℕ → α} {init : Fin E.order → α} (h : E.IsSolution u)
    (heq : ∀ n : Fin E.order, u n = init n) : ∀ n, u n = E.mkSol init n := by
  /-
    α : Type u_1
    inst✝ : CommSemiring α
    E : LinearRecurrence α
    u : Nat → α
    init : Fin E.order → α
    h : E.IsSolution u
    heq : ∀ (n : Fin E.order), Eq (u ↑n) (init n)
    ⊢ ∀ (n : Nat), Eq (u n) (E.mkSol init n)
  -/
  intro n
  /-
    α : Type u_1
    inst✝ : CommSemiring α
    E : LinearRecurrence α
    u : Nat → α
    init : Fin E.order → α
    h : E.IsSolution u
    heq : ∀ (n : Fin E.order), Eq (u ↑n) (init n)
    n : Nat
    ⊢ Eq (u n) (E.mkSol init n)
  -/
  rw [mkSol]
  /-
    α : Type u_1
    inst✝ : CommSemiring α
    E : LinearRecurrence α
    u : Nat → α
    init : Fin E.order → α
    h : E.IsSolution u
    heq : ∀ (n : Fin E.order), Eq (u ↑n) (init n)
    n : Nat
    ⊢ Eq (u n) (dite (LT.lt n E.order) (fun h => init ⟨n, h⟩) fun h => Finset.univ …
  -/
  split_ifs with h'
    /-
      case pos
      α : Type u_1
      inst✝ : CommSemiring α
      E : LinearRecurrence α
      u : Nat → α
      init : Fin E.order → α
      h : E.IsSolution u
      heq : ∀ (n : Fin E.order), Eq (u ↑n) (init n)
      n : Nat
      h' : LT.lt n E.order
      ⊢ Eq (u n) (init ⟨n, ⋯⟩)
    -/
  · exact mod_cast heq ⟨n, h'⟩
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    inst✝ : CommSemiring α
    E : LinearRecurrence α
    u : Nat → α
    init : Fin E.order → α
    h : E.IsSolution u
    heq : ∀ (n : Fin E.order), Eq (u ↑n) (init n)
    n : Nat
    h' : Not (LT.lt n E.order)
    ⊢ Eq (u n) (Finset.univ.sum fun x => HMul.hMul (E.coeffs x) (E.mkSol init (HAd …
  -/
  rw [← tsub_add_cancel_of_le (le_of_not_lt h'), h (n - E.order)]
  /-
    case neg
    α : Type u_1
    inst✝ : CommSemiring α
    E : LinearRecurrence α
    u : Nat → α
    init : Fin E.order → α
    h : E.IsSolution u
    heq : ∀ (n : Fin E.order), Eq (u ↑n) (init n)
    n : Nat
    h' : Not (LT.lt n E.order)
    ⊢ Eq (Finset.univ.sum fun i => HMul.hMul (E.coeffs i) (u (HAdd.hAdd (HSub.hSub …
  -/
  congr with k
  /-
    case neg.e_f.h
    α : Type u_1
    inst✝ : CommSemiring α
    E : LinearRecurrence α
    u : Nat → α
    init : Fin E.order → α
    h : E.IsSolution u
    heq : ∀ (n : Fin E.order), Eq (u ↑n) (init n)
    n : Nat
    h' : Not (LT.lt n E.order)
    k : Fin E.order
    ⊢ Eq (HMul.hMul (E.coeffs k) (u (HAdd.hAdd (HSub.hSub n E.order) ↑k))) (HMul.h …
  -/
  have : n - E.order + k < n := by omega
  /-
    case neg.e_f.h
    α : Type u_1
    inst✝ : CommSemiring α
    E : LinearRecurrence α
    u : Nat → α
    init : Fin E.order → α
    h : E.IsSolution u
    heq : ∀ (n : Fin E.order), Eq (u ↑n) (init n)
    n : Nat
    h' : Not (LT.lt n E.order)
    k : Fin E.order
    this : LT.lt (HAdd.hAdd (HSub.hSub n E.order) ↑k) n
    ⊢ Eq (HMul.hMul (E.coeffs k) (u (HAdd.hAdd (HSub.hSub n E.order) ↑k))) (HMul.h …
  -/
  rw [eq_mk_of_is_sol_of_eq_init h heq (n - E.order + k)]
  /-
    case neg.e_f.h
    α : Type u_1
    inst✝ : CommSemiring α
    E : LinearRecurrence α
    u : Nat → α
    init : Fin E.order → α
    h : E.IsSolution u
    heq : ∀ (n : Fin E.order), Eq (u ↑n) (init n)
    n : Nat
    h' : Not (LT.lt n E.order)
    k : Fin E.order
    this : LT.lt (HAdd.hAdd (HSub.hSub n E.order) ↑k) n
    ⊢ Eq (HMul.hMul (E.coeffs k) (E.mkSol init (HAdd.hAdd (HSub.hSub n E.order) ↑k …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- If `u` is a solution to `E` and `init` designates its first `E.order` values,
  then `u = E.mkSol init`. This proves that `E.mkSol init` is the only solution
  of `E` whose first `E.order` values are given by `init`. -/
theorem eq_mk_of_is_sol_of_eq_init' {u : ℕ → α} {init : Fin E.order → α} (h : E.IsSolution u)
    (heq : ∀ n : Fin E.order, u n = init n) : u = E.mkSol init :=
  funext (E.eq_mk_of_is_sol_of_eq_init h heq)


/-- The space of solutions of `E`, as a `Submodule` over `α` of the module `ℕ → α`. -/
def solSpace : Submodule α (ℕ → α) where
  carrier := { u | E.IsSolution u }
                    /-
                      α : Type u_1
                      inst✝ : CommSemiring α
                      E : LinearRecurrence α
                      n : Nat
                      ⊢ Eq (0 (HAdd.hAdd n E.order)) (Finset.univ.sum fun i => HMul.hMul (E.coeffs i …
                    -/
                               /-
                                 α : Type u_1
                                 inst✝ : CommSemiring α
                                 E : LinearRecurrence α
                                 u v : Nat → α
                                 hu : Membership.mem (setOf fun u => E.IsSolution u) u
                                 hv : Membership.mem (setOf fun u => E.IsSolution u) v
                                 n : Nat
                                 ⊢ Eq (HAdd.hAdd u v (HAdd.hAdd n E.order)) (Finset.univ.sum fun i => HMul.hMul …
                               -/
  zero_mem' n := by simp
                               /-
                                 🎉 no goals
                               -/
                    /-
                      🎉 no goals
                    -/
  add_mem' {u v} hu hv n := by simp [mul_add, sum_add_distrib, hu n, hv n]
                           /-
                             α : Type u_1
                             inst✝ : CommSemiring α
                             E : LinearRecurrence α
                             a : α
                             u : Nat → α
                             hu : Membership.mem { carrier := setOf fun u => E.IsSolution u, add_mem' := ⋯, …
                             n : Nat
                             ⊢ Eq (HSMul.hSMul a u (HAdd.hAdd n E.order)) (Finset.univ.sum fun i => HMul.hM …
                           -/
  smul_mem' a u hu n := by simp [hu n, mul_sum]; congr; ext; ac_rfl
                                                             /-
                                                               🎉 no goals
                                                             -/


/-- Defining property of the solution space : `u` is a solution
  iff it belongs to the solution space. -/
theorem is_sol_iff_mem_solSpace (u : ℕ → α) : E.IsSolution u ↔ u ∈ E.solSpace :=
  Iff.rfl


/-- The function that maps a solution `u` of `E` to its first
  `E.order` terms as a `LinearEquiv`. -/
def toInit : E.solSpace ≃ₗ[α] Fin E.order → α where
  toFun u x := (u : ℕ → α) x
  map_add' u v := by
    /-
      α : Type u_1
      inst✝ : CommSemiring α
      E : LinearRecurrence α
      u v : Subtype fun x => Membership.mem E.solSpace x
      ⊢ Eq ((fun u x => ↑u ↑x) (HAdd.hAdd u v)) (HAdd.hAdd ((fun u x => ↑u ↑x) u) (( …
    -/
    ext
    /-
      case h
      α : Type u_1
      inst✝ : CommSemiring α
      E : LinearRecurrence α
      u v : Subtype fun x => Membership.mem E.solSpace x
      x✝ : Fin E.order
      ⊢ Eq ((fun u x => ↑u ↑x) (HAdd.hAdd u v) x✝) (HAdd.hAdd ((fun u x => ↑u ↑x) u) …
    -/
    simp
    /-
      🎉 no goals
    -/
  map_smul' a u := by
    /-
      α : Type u_1
      inst✝ : CommSemiring α
      E : LinearRecurrence α
      a : α
      u : Subtype fun x => Membership.mem E.solSpace x
      ⊢ Eq ({ toFun := fun u x => ↑u ↑x, map_add' := ⋯ }.toFun (HSMul.hSMul a u)) (H …
    -/
    ext
    /-
      case h
      α : Type u_1
      inst✝ : CommSemiring α
      E : LinearRecurrence α
      a : α
      u : Subtype fun x => Membership.mem E.solSpace x
      x✝ : Fin E.order
      ⊢ Eq ({ toFun := fun u x => ↑u ↑x, map_add' := ⋯ }.toFun (HSMul.hSMul a u) x✝) …
    -/
    simp
    /-
      🎉 no goals
    -/
  invFun u := ⟨E.mkSol u, E.is_sol_mkSol u⟩
                   /-
                     α : Type u_1
                     inst✝ : CommSemiring α
                     E : LinearRecurrence α
                     u : Subtype fun x => Membership.mem E.solSpace x
                     ⊢ Eq ((fun u => ⟨E.mkSol u, ⋯⟩) ({ toFun := fun u x => ↑u ↑x, map_add' := ⋯, m …
                   -/
  left_inv u := by ext n; symm; apply E.eq_mk_of_is_sol_of_eq_init u.2; intro k; rfl
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/
  right_inv u := funext_iff.mpr fun n ↦ E.mkSol_eq_init u n


/-- Two solutions are equal iff they are equal on `range E.order`. -/
theorem sol_eq_of_eq_init (u v : ℕ → α) (hu : E.IsSolution u) (hv : E.IsSolution v) :
    u = v ↔ Set.EqOn u v ↑(range E.order) := by
  /-
    α : Type u_1
    inst✝ : CommSemiring α
    E : LinearRecurrence α
    u v : Nat → α
    hu : E.IsSolution u
    hv : E.IsSolution v
    ⊢ Iff (Eq u v) (Set.EqOn u v ↑(Finset.range E.order))
  -/
  refine Iff.intro (fun h x _ ↦ h ▸ rfl) ?_
  /-
    α : Type u_1
    inst✝ : CommSemiring α
    E : LinearRecurrence α
    u v : Nat → α
    hu : E.IsSolution u
    hv : E.IsSolution v
    ⊢ Set.EqOn u v ↑(Finset.range E.order) → Eq u v
  -/
  intro h
  /-
    α : Type u_1
    inst✝ : CommSemiring α
    E : LinearRecurrence α
    u v : Nat → α
    hu : E.IsSolution u
    hv : E.IsSolution v
    h : Set.EqOn u v ↑(Finset.range E.order)
    ⊢ Eq u v
  -/
  set u' : ↥E.solSpace := ⟨u, hu⟩
  /-
    α : Type u_1
    inst✝ : CommSemiring α
    E : LinearRecurrence α
    u v : Nat → α
    hu : E.IsSolution u
    hv : E.IsSolution v
    h : Set.EqOn u v ↑(Finset.range E.order)
    u' : Subtype fun x => Membership.mem E.solSpace x := ⟨u, hu⟩
    ⊢ Eq u v
  -/
  set v' : ↥E.solSpace := ⟨v, hv⟩
  /-
    α : Type u_1
    inst✝ : CommSemiring α
    E : LinearRecurrence α
    u v : Nat → α
    hu : E.IsSolution u
    hv : E.IsSolution v
    h : Set.EqOn u v ↑(Finset.range E.order)
    u' : Subtype fun x => Membership.mem E.solSpace x := ⟨u, hu⟩
    v' : Subtype fun x => Membership.mem E.solSpace x := ⟨v, hv⟩
    ⊢ Eq u v
  -/
  change u'.val = v'.val
  /-
    α : Type u_1
    inst✝ : CommSemiring α
    E : LinearRecurrence α
    u v : Nat → α
    hu : E.IsSolution u
    hv : E.IsSolution v
    h : Set.EqOn u v ↑(Finset.range E.order)
    u' : Subtype fun x => Membership.mem E.solSpace x := ⟨u, hu⟩
    v' : Subtype fun x => Membership.mem E.solSpace x := ⟨v, hv⟩
    ⊢ Eq ↑u' ↑v'
  -/
  suffices h' : u' = v' from h' ▸ rfl
  /-
    α : Type u_1
    inst✝ : CommSemiring α
    E : LinearRecurrence α
    u v : Nat → α
    hu : E.IsSolution u
    hv : E.IsSolution v
    h : Set.EqOn u v ↑(Finset.range E.order)
    u' : Subtype fun x => Membership.mem E.solSpace x := ⟨u, hu⟩
    v' : Subtype fun x => Membership.mem E.solSpace x := ⟨v, hv⟩
    ⊢ Eq u' v'
  -/
  rw [← E.toInit.toEquiv.apply_eq_iff_eq, LinearEquiv.coe_toEquiv]
  /-
    α : Type u_1
    inst✝ : CommSemiring α
    E : LinearRecurrence α
    u v : Nat → α
    hu : E.IsSolution u
    hv : E.IsSolution v
    h : Set.EqOn u v ↑(Finset.range E.order)
    u' : Subtype fun x => Membership.mem E.solSpace x := ⟨u, hu⟩
    v' : Subtype fun x => Membership.mem E.solSpace x := ⟨v, hv⟩
    ⊢ Eq (E.toInit u') (E.toInit v')
  -/
  ext x
  /-
    case h
    α : Type u_1
    inst✝ : CommSemiring α
    E : LinearRecurrence α
    u v : Nat → α
    hu : E.IsSolution u
    hv : E.IsSolution v
    h : Set.EqOn u v ↑(Finset.range E.order)
    u' : Subtype fun x => Membership.mem E.solSpace x := ⟨u, hu⟩
    v' : Subtype fun x => Membership.mem E.solSpace x := ⟨v, hv⟩
    x : Fin E.order
    ⊢ Eq (E.toInit u' x) (E.toInit v' x)
  -/
  exact mod_cast h (mem_range.mpr x.2)
  /-
    🎉 no goals
  -/


/-- `E.tupleSucc` maps `![s₀, s₁, ..., sₙ]` to `![s₁, ..., sₙ, ∑ (E.coeffs i) * sᵢ]`,
  where `n := E.order`. -/
def tupleSucc : (Fin E.order → α) →ₗ[α] Fin E.order → α where
  toFun X i := if h : (i : ℕ) + 1 < E.order then X ⟨i + 1, h⟩ else ∑ i, E.coeffs i * X i
  map_add' x y := by
    /-
      α : Type u_1
      inst✝ : CommSemiring α
      E : LinearRecurrence α
      x y : Fin E.order → α
      ⊢ Eq ((fun X i => dite (LT.lt (HAdd.hAdd (↑i) 1) E.order) (fun h => X ⟨HAdd.hA …
    -/
    ext i
    /-
      case h
      α : Type u_1
      inst✝ : CommSemiring α
      E : LinearRecurrence α
      x y : Fin E.order → α
      i : Fin E.order
      ⊢ Eq ((fun X i => dite (LT.lt (HAdd.hAdd (↑i) 1) E.order) (fun h => X ⟨HAdd.hA …
    -/
    simp only
    /-
      case h
      α : Type u_1
      inst✝ : CommSemiring α
      E : LinearRecurrence α
      x y : Fin E.order → α
      i : Fin E.order
      ⊢ Eq (dite (LT.lt (HAdd.hAdd (↑i) 1) E.order) (fun h => HAdd.hAdd x y ⟨HAdd.hA …
    -/
                         /-
                           🎉 no goals
                         -/
    split_ifs with h <;> simp [h, mul_add, sum_add_distrib]
                         /-
                           🎉 no goals
                         -/
  map_smul' x y := by
    /-
      α : Type u_1
      inst✝ : CommSemiring α
      E : LinearRecurrence α
      x : α
      y : Fin E.order → α
      ⊢ Eq ({ toFun := fun X i => dite (LT.lt (HAdd.hAdd (↑i) 1) E.order) (fun h =>  …
    -/
    ext i
    /-
      case h
      α : Type u_1
      inst✝ : CommSemiring α
      E : LinearRecurrence α
      x : α
      y : Fin E.order → α
      i : Fin E.order
      ⊢ Eq ({ toFun := fun X i => dite (LT.lt (HAdd.hAdd (↑i) 1) E.order) (fun h =>  …
    -/
    simp only
    /-
      case h
      α : Type u_1
      inst✝ : CommSemiring α
      E : LinearRecurrence α
      x : α
      y : Fin E.order → α
      i : Fin E.order
      ⊢ Eq (dite (LT.lt (HAdd.hAdd (↑i) 1) E.order) (fun h => HSMul.hSMul x y ⟨HAdd. …
    -/
                         /-
                           🎉 no goals
                         -/
    split_ifs with h <;> simp [h, mul_sum]
    /-
      case neg
      α : Type u_1
      inst✝ : CommSemiring α
      E : LinearRecurrence α
      x : α
      y : Fin E.order → α
      i : Fin E.order
      h : Not (LT.lt (HAdd.hAdd (↑i) 1) E.order)
      ⊢ Eq (Finset.univ.sum fun x_1 => HMul.hMul (E.coeffs x_1) (HMul.hMul x (y x_1) …
    -/
    exact sum_congr rfl fun x _ ↦ by ac_rfl
    /-
      🎉 no goals
    -/


/-- The dimension of `E.solSpace` is `E.order`. -/
theorem solSpace_rank : Module.rank α E.solSpace = E.order :=
  letI := nontrivial_of_invariantBasisNumber α
  @rank_fin_fun α _ _ E.order ▸ E.toInit.rank_eq


/-- The characteristic polynomial of `E` is
`X ^ E.order - ∑ i : Fin E.order, (E.coeffs i) * X ^ i`. -/
def charPoly : α[X] :=
  Polynomial.monomial E.order 1 - ∑ i : Fin E.order, Polynomial.monomial i (E.coeffs i)


/-- The geometric sequence `q^n` is a solution of `E` iff
  `q` is a root of `E`'s characteristic polynomial. -/
theorem geom_sol_iff_root_charPoly (q : α) :
    (E.IsSolution fun n ↦ q ^ n) ↔ E.charPoly.IsRoot q := by
  /-
    α : Type u_1
    inst✝ : CommRing α
    E : LinearRecurrence α
    q : α
    ⊢ Iff (E.IsSolution fun n => HPow.hPow q n) (E.charPoly.IsRoot q)
  -/
  rw [charPoly, Polynomial.IsRoot.def, Polynomial.eval]
  simp only [Polynomial.eval₂_finset_sum, one_mul, RingHom.id_apply, Polynomial.eval₂_monomial,
    Polynomial.eval₂_sub]
  /-
    α : Type u_1
    inst✝ : CommRing α
    E : LinearRecurrence α
    q : α
    ⊢ Iff (E.IsSolution fun n => HPow.hPow q n) (Eq (HSub.hSub (HPow.hPow q E.orde …
  -/
  constructor
    /-
      case mp
      α : Type u_1
      inst✝ : CommRing α
      E : LinearRecurrence α
      q : α
      ⊢ (E.IsSolution fun n => HPow.hPow q n) → Eq (HSub.hSub (HPow.hPow q E.order)  …
    -/
  · intro h
    /-
      case mp
      α : Type u_1
      inst✝ : CommRing α
      E : LinearRecurrence α
      q : α
      h : E.IsSolution fun n => HPow.hPow q n
      ⊢ Eq (HSub.hSub (HPow.hPow q E.order) (Finset.univ.sum fun x => HMul.hMul (E.c …
    -/
    simpa [sub_eq_zero] using h 0
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u_1
      inst✝ : CommRing α
      E : LinearRecurrence α
      q : α
      ⊢ Eq (HSub.hSub (HPow.hPow q E.order) (Finset.univ.sum fun x => HMul.hMul (E.c …
    -/
  · intro h n
    /-
      case mpr
      α : Type u_1
      inst✝ : CommRing α
      E : LinearRecurrence α
      q : α
      h : Eq (HSub.hSub (HPow.hPow q E.order) (Finset.univ.sum fun x => HMul.hMul (E …
      n : Nat
      ⊢ Eq ((fun n => HPow.hPow q n) (HAdd.hAdd n E.order)) (Finset.univ.sum fun i = …
    -/
    simp only [pow_add, sub_eq_zero.mp h, mul_sum]
    /-
      case mpr
      α : Type u_1
      inst✝ : CommRing α
      E : LinearRecurrence α
      q : α
      h : Eq (HSub.hSub (HPow.hPow q E.order) (Finset.univ.sum fun x => HMul.hMul (E …
      n : Nat
      ⊢ Eq (Finset.univ.sum fun i => HMul.hMul (HPow.hPow q n) (HMul.hMul (E.coeffs  …
    -/
    exact sum_congr rfl fun _ _ ↦ by ring
    /-
      🎉 no goals
    -/


