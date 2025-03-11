/-- Given a finite set of monomials `monoms : ι → Finset (κ →₀ ℕ)`, the
`genericPolyMap monoms` is an indexed collection of elements of the `FreeCommRing`,
that can be evaluated to any collection `p : ι → MvPolynomial κ R` of
polynomials such that `∀ i, (p i).support ⊆ monoms i`. -/
def genericPolyMap (monoms : ι → Finset (κ →₀ ℕ)) :
    ι → FreeCommRing ((Σ i : ι, monoms i) ⊕ κ) :=
  fun i => (monoms i).attach.sum
    (fun m => FreeCommRing.of (Sum.inl ⟨i, m⟩) *
      Finsupp.prod m.1 (fun j n => FreeCommRing.of (Sum.inr j)^ n))


/-- Collections of `MvPolynomial`s, `p : ι → MvPolynomial κ R` such
that `∀ i, (p i).support ⊆ monoms i` can be identified with functions
`(Σ i, monoms i) → R` by using the coefficient function -/
noncomputable def mvPolynomialSupportLEEquiv
    [DecidableEq κ] [CommRing R] [DecidableEq R]
    (monoms : ι → Finset (κ →₀ ℕ)) :
    { p : ι → MvPolynomial κ R // ∀ i, (p i).support ⊆ monoms i } ≃
      ((Σ i, monoms i) → R) :=
  { toFun := fun p i => (p.1 i.1).coeff i.2,
    invFun := fun p => ⟨fun i =>
      { toFun := fun m => if hm : m ∈ monoms i then p ⟨i, ⟨m, hm⟩⟩ else 0
        support := (monoms i).filter (fun m => ∃ hm : m ∈ monoms i, p ⟨i, ⟨m, hm⟩⟩ ≠ 0),
                                /-
                                  ι : Type u_1
                                  κ : Type u_2
                                  R : Type u_3
                                  inst✝² : DecidableEq κ
                                  inst✝¹ : CommRing R
                                  inst✝ : DecidableEq R
                                  monoms : ι → Finset (Finsupp κ Nat)
                                  p : (Sigma fun i => Subtype fun x => Membership.mem (monoms i) x) → R
                                  i : ι
                                  ⊢ ∀ (a : Finsupp κ Nat), Iff (Membership.mem (Finset.filter (fun m => Exists f …
                                -/
        mem_support_toFun := by simp (config := {contextual := true}) },
                                /-
                                  🎉 no goals
                                -/
      fun i => Finset.filter_subset _ _⟩,
    left_inv := fun p => by
      /-
        ι : Type u_1
        κ : Type u_2
        R : Type u_3
        inst✝² : DecidableEq κ
        inst✝¹ : CommRing R
        inst✝ : DecidableEq R
        monoms : ι → Finset (Finsupp κ Nat)
        p : Subtype fun p => ∀ (i : ι), HasSubset.Subset (p i).support (monoms i)
        ⊢ Eq ((fun p => ⟨fun i => { support := Finset.filter (fun m => Exists fun hm = …
      -/
      ext i m
      /-
        case a.h.a
        ι : Type u_1
        κ : Type u_2
        R : Type u_3
        inst✝² : DecidableEq κ
        inst✝¹ : CommRing R
        inst✝ : DecidableEq R
        monoms : ι → Finset (Finsupp κ Nat)
        p : Subtype fun p => ∀ (i : ι), HasSubset.Subset (p i).support (monoms i)
        i : ι
        m : Finsupp κ Nat
        ⊢ Eq (MvPolynomial.coeff m (↑((fun p => ⟨fun i => { support := Finset.filter ( …
      -/
      simp only [coeff, ne_eq, exists_prop, dite_eq_ite, Finsupp.coe_mk, ite_eq_left_iff]
      /-
        case a.h.a
        ι : Type u_1
        κ : Type u_2
        R : Type u_3
        inst✝² : DecidableEq κ
        inst✝¹ : CommRing R
        inst✝ : DecidableEq R
        monoms : ι → Finset (Finsupp κ Nat)
        p : Subtype fun p => ∀ (i : ι), HasSubset.Subset (p i).support (monoms i)
        i : ι
        m : Finsupp κ Nat
        ⊢ Not (Membership.mem (monoms i) m) → Eq 0 ((↑p i) m)
      -/
      intro hm
      /-
        case a.h.a
        ι : Type u_1
        κ : Type u_2
        R : Type u_3
        inst✝² : DecidableEq κ
        inst✝¹ : CommRing R
        inst✝ : DecidableEq R
        monoms : ι → Finset (Finsupp κ Nat)
        p : Subtype fun p => ∀ (i : ι), HasSubset.Subset (p i).support (monoms i)
        i : ι
        m : Finsupp κ Nat
        hm : Not (Membership.mem (monoms i) m)
        ⊢ Eq 0 ((↑p i) m)
      -/
      have : m ∉ (p.1 i).support := fun h => hm (p.2 i h)
      /-
        case a.h.a
        ι : Type u_1
        κ : Type u_2
        R : Type u_3
        inst✝² : DecidableEq κ
        inst✝¹ : CommRing R
        inst✝ : DecidableEq R
        monoms : ι → Finset (Finsupp κ Nat)
        p : Subtype fun p => ∀ (i : ι), HasSubset.Subset (p i).support (monoms i)
        i : ι
        m : Finsupp κ Nat
        hm : Not (Membership.mem (monoms i) m)
        this : Not (Membership.mem (↑p i).support m)
        ⊢ Eq 0 ((↑p i) m)
      -/
      simpa [coeff, eq_comm, MvPolynomial.mem_support_iff] using this
      /-
        🎉 no goals
      -/
                             /-
                               ι : Type u_1
                               κ : Type u_2
                               R : Type u_3
                               inst✝² : DecidableEq κ
                               inst✝¹ : CommRing R
                               inst✝ : DecidableEq R
                               monoms : ι → Finset (Finsupp κ Nat)
                               p : (Sigma fun i => Subtype fun x => Membership.mem (monoms i) x) → R
                               ⊢ Eq ((fun p i => MvPolynomial.coeff (↑i.snd) (↑p i.fst)) ((fun p => ⟨fun i => …
                             -/
    right_inv := fun p => by ext; simp [coeff] }
                                  /-
                                    🎉 no goals
                                  -/


@[simp]
theorem MvPolynomialSupportLEEquiv_symm_apply_coeff [DecidableEq κ] [CommRing R] [DecidableEq R]
    (p : ι → MvPolynomial κ R) : (mvPolynomialSupportLEEquiv (fun i => (p i).support)).symm
      (fun i => (p i.1).coeff i.2.1) = ⟨p, fun _ => Finset.Subset.refl _⟩ :=
  (mvPolynomialSupportLEEquiv (R := R) (fun i : ι => (p i).support)).symm_apply_apply
    ⟨p, fun _ => Finset.Subset.refl _⟩


@[simp]
theorem lift_genericPolyMap [DecidableEq κ] [CommRing R]
    [DecidableEq R] (monoms : ι → Finset (κ →₀ ℕ))
    (f : (i : ι) × { x // x ∈ monoms i } ⊕ κ → R) (i : ι) :
    FreeCommRing.lift f (genericPolyMap monoms i) =
      MvPolynomial.eval (f ∘ Sum.inr)
        (((mvPolynomialSupportLEEquiv monoms).symm
          (f ∘ Sum.inl)).1 i) := by
  simp only [genericPolyMap, Finsupp.prod_pow, map_sum, map_mul, lift_of, support,
    mvPolynomialSupportLEEquiv, coeff, map_prod, Finset.sum_filter, MvPolynomial.eval_eq,
    ne_eq, Function.comp, Equiv.coe_fn_symm_mk, Finsupp.coe_mk]
  /-
    ι : Type u_1
    κ : Type u_2
    R : Type u_3
    inst✝² : DecidableEq κ
    inst✝¹ : CommRing R
    inst✝ : DecidableEq R
    monoms : ι → Finset (Finsupp κ Nat)
    f : Sum (Sigma fun i => Subtype fun x => Membership.mem (monoms i) x) κ → R
    i : ι
    ⊢ Eq ((monoms i).attach.sum fun x => HMul.hMul (f (Sum.inl ⟨i, x⟩)) ((FreeComm …
  -/
  conv_rhs => rw [← Finset.sum_attach]
  /-
    ι : Type u_1
    κ : Type u_2
    R : Type u_3
    inst✝² : DecidableEq κ
    inst✝¹ : CommRing R
    inst✝ : DecidableEq R
    monoms : ι → Finset (Finsupp κ Nat)
    f : Sum (Sigma fun i => Subtype fun x => Membership.mem (monoms i) x) κ → R
    i : ι
    ⊢ Eq ((monoms i).attach.sum fun x => HMul.hMul (f (Sum.inl ⟨i, x⟩)) ((FreeComm …
  -/
  refine Finset.sum_congr rfl ?_
  /-
    ι : Type u_1
    κ : Type u_2
    R : Type u_3
    inst✝² : DecidableEq κ
    inst✝¹ : CommRing R
    inst✝ : DecidableEq R
    monoms : ι → Finset (Finsupp κ Nat)
    f : Sum (Sigma fun i => Subtype fun x => Membership.mem (monoms i) x) κ → R
    i : ι
    ⊢ ∀ (x : Subtype fun x => Membership.mem (monoms i) x), Membership.mem (monoms …
  -/
  intros m _
  simp only [Finsupp.prod, map_prod, map_pow, lift_of, Subtype.coe_eta, Finset.coe_mem,
    exists_prop, true_and, dite_eq_ite, ite_true, ite_not]
  /-
    ι : Type u_1
    κ : Type u_2
    R : Type u_3
    inst✝² : DecidableEq κ
    inst✝¹ : CommRing R
    inst✝ : DecidableEq R
    monoms : ι → Finset (Finsupp κ Nat)
    f : Sum (Sigma fun i => Subtype fun x => Membership.mem (monoms i) x) κ → R
    i : ι
    m : Subtype fun x => Membership.mem (monoms i) x
    a✝ : Membership.mem (monoms i).attach m
    ⊢ Eq (HMul.hMul (f (Sum.inl ⟨i, m⟩)) ((↑m).support.prod fun x => HPow.hPow (f  …
  -/
                        /-
                          🎉 no goals
                        -/
  split_ifs with h0 <;> simp_all
                        /-
                          🎉 no goals
                        -/


