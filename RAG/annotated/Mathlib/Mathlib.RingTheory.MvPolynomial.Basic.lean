instance [CharP R p] : CharP (MvPolynomial σ R) p where
                            /-
                              σ : Type u
                              R : Type v
                              inst✝¹ : CommSemiring R
                              p m : Nat
                              inst✝ : CharP R p
                              n : Nat
                              ⊢ Iff (Eq (↑n) 0) (Dvd.dvd p n)
                            -/
  cast_eq_zero_iff' n := by rw [← C_eq_coe_nat, ← C_0, C_inj, CharP.cast_eq_zero_iff R p]
                            /-
                              🎉 no goals
                            -/


instance [CharZero R] : CharZero (MvPolynomial σ R) where
                               /-
                                 σ : Type u
                                 R : Type v
                                 inst✝¹ : CommSemiring R
                                 p m : Nat
                                 inst✝ : CharZero R
                                 x y : Nat
                                 hxy : Eq ↑x ↑y
                                 ⊢ Eq x y
                               -/
  cast_injective x y hxy := by rwa [← C_eq_coe_nat, ← C_eq_coe_nat, C_inj, Nat.cast_inj] at hxy
                               /-
                                 🎉 no goals
                               -/


theorem mapRange_eq_map {R S : Type*} [CommSemiring R] [CommSemiring S] (p : MvPolynomial σ R)
    (f : R →+* S) : Finsupp.mapRange f f.map_zero p = map f p := by
  /-
    σ : Type u
    R : Type u_1
    S : Type u_2
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S
    p : MvPolynomial σ R
    f : RingHom R S
    ⊢ Eq (Finsupp.mapRange ⇑f ⋯ p) ((MvPolynomial.map f) p)
  -/
  rw [p.as_sum, Finsupp.mapRange_finset_sum, map_sum (map f)]
  /-
    σ : Type u
    R : Type u_1
    S : Type u_2
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S
    p : MvPolynomial σ R
    f : RingHom R S
    ⊢ Eq (p.support.sum fun x => Finsupp.mapRange ⇑f ⋯ ((MvPolynomial.monomial x)  …
  -/
  refine Finset.sum_congr rfl fun n _ => ?_
  /-
    σ : Type u
    R : Type u_1
    S : Type u_2
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S
    p : MvPolynomial σ R
    f : RingHom R S
    n : Finsupp σ Nat
    x✝ : Membership.mem p.support n
    ⊢ Eq (Finsupp.mapRange ⇑f ⋯ ((MvPolynomial.monomial n) (MvPolynomial.coeff n p …
  -/
  rw [map_monomial, ← single_eq_monomial, Finsupp.mapRange_single, single_eq_monomial]
  /-
    🎉 no goals
  -/


/-- The submodule of polynomials that are sum of monomials in the set `s`. -/
def restrictSupport (s : Set (σ →₀ ℕ)) : Submodule R (MvPolynomial σ R) :=
  Finsupp.supported _ _ s


/-- `restrictSupport R s` has a canonical `R`-basis indexed by `s`. -/
def basisRestrictSupport (s : Set (σ →₀ ℕ)) : Basis s R (restrictSupport R s) where
  repr := Finsupp.supportedEquivFinsupp s


theorem restrictSupport_mono {s t : Set (σ →₀ ℕ)} (h : s ⊆ t) :
    restrictSupport R s ≤ restrictSupport R t := Finsupp.supported_mono h


/-- The submodule of polynomials of total degree less than or equal to `m`. -/
def restrictTotalDegree (m : ℕ) : Submodule R (MvPolynomial σ R) :=
  restrictSupport R { n | (n.sum fun _ e => e) ≤ m }


/-- The submodule of polynomials such that the degree with respect to each individual variable is
less than or equal to `m`. -/
def restrictDegree (m : ℕ) : Submodule R (MvPolynomial σ R) :=
  restrictSupport R { n | ∀ i, n i ≤ m }


theorem mem_restrictTotalDegree (p : MvPolynomial σ R) :
    p ∈ restrictTotalDegree σ R m ↔ p.totalDegree ≤ m := by
  /-
    σ : Type u
    R : Type v
    inst✝ : CommSemiring R
    m : Nat
    p : MvPolynomial σ R
    ⊢ Iff (Membership.mem (MvPolynomial.restrictTotalDegree σ R m) p) (LE.le p.tot …
  -/
  rw [totalDegree, Finset.sup_le_iff]
  /-
    σ : Type u
    R : Type v
    inst✝ : CommSemiring R
    m : Nat
    p : MvPolynomial σ R
    ⊢ Iff (Membership.mem (MvPolynomial.restrictTotalDegree σ R m) p) (∀ (b : Fins …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem mem_restrictDegree (p : MvPolynomial σ R) (n : ℕ) :
    p ∈ restrictDegree σ R n ↔ ∀ s ∈ p.support, ∀ i, (s : σ →₀ ℕ) i ≤ n := by
  /-
    σ : Type u
    R : Type v
    inst✝ : CommSemiring R
    p : MvPolynomial σ R
    n : Nat
    ⊢ Iff (Membership.mem (MvPolynomial.restrictDegree σ R n) p) (∀ (s : Finsupp σ …
  -/
  rw [restrictDegree, restrictSupport, Finsupp.mem_supported]
  /-
    σ : Type u
    R : Type v
    inst✝ : CommSemiring R
    p : MvPolynomial σ R
    n : Nat
    ⊢ Iff (HasSubset.Subset (↑p.support) (setOf fun n_1 => ∀ (i : σ), LE.le (n_1 i …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem mem_restrictDegree_iff_sup [DecidableEq σ] (p : MvPolynomial σ R) (n : ℕ) :
    p ∈ restrictDegree σ R n ↔ ∀ i, p.degrees.count i ≤ n := by
  simp only [mem_restrictDegree, degrees_def, Multiset.count_finset_sup, Finsupp.count_toMultiset,
    Finset.sup_le_iff]
  /-
    σ : Type u
    R : Type v
    inst✝¹ : CommSemiring R
    inst✝ : DecidableEq σ
    p : MvPolynomial σ R
    n : Nat
    ⊢ Iff (∀ (s : Finsupp σ Nat), Membership.mem p.support s → ∀ (i : σ), LE.le (s …
  -/
  exact ⟨fun h n s hs => h s hs n, fun h s hs n => h n s hs⟩
  /-
    🎉 no goals
  -/


theorem restrictTotalDegree_le_restrictDegree (m : ℕ) :
    restrictTotalDegree σ R m ≤ restrictDegree σ R m :=
  fun p hp ↦ (mem_restrictDegree _ _ _).mpr fun s hs i ↦ (degreeOf_le_iff.mp
    (degreeOf_le_totalDegree p i) s hs).trans ((mem_restrictTotalDegree _ _ _).mp hp)


/-- The monomials form a basis on `MvPolynomial σ R`. -/
def basisMonomials : Basis (σ →₀ ℕ) R (MvPolynomial σ R) :=
  Finsupp.basisSingleOne


@[simp]
theorem coe_basisMonomials :
    (basisMonomials σ R : (σ →₀ ℕ) → MvPolynomial σ R) = fun s => monomial s 1 :=
  rfl


/-- The `R`-module `MvPolynomial σ R` is free. -/
instance : Module.Free R (MvPolynomial σ R) :=
  Module.Free.of_basis (MvPolynomial.basisMonomials σ R)


theorem linearIndependent_X : LinearIndependent R (X : σ → MvPolynomial σ R) :=
  (basisMonomials σ R).linearIndependent.comp (fun s : σ => Finsupp.single s 1)
    (Finsupp.single_left_injective one_ne_zero)


private lemma finite_setOf_bounded (α) [Finite α] (n : ℕ) : Finite {f : α →₀ ℕ | ∀ a, f a ≤ n} :=
  ((Set.Finite.pi' fun _ ↦ Set.finite_le_nat _).preimage DFunLike.coe_injective.injOn).to_subtype


instance [Finite σ] (N : ℕ) : Module.Finite R (restrictDegree σ R N) :=
  have := finite_setOf_bounded σ N
  Module.Finite.of_basis (basisRestrictSupport R _)


instance [Finite σ] (N : ℕ) : Module.Finite R (restrictTotalDegree σ R N) :=
  have := finite_setOf_bounded σ N
  have : Finite {s : σ →₀ ℕ | s.sum (fun _ e ↦ e) ≤ N} := by
    /-
      σ : Type u
      R : Type v
      inst✝¹ : CommSemiring R
      p m : Nat
      inst✝ : Finite σ
      N : Nat
      this : Finite ↑(setOf fun f => ∀ (a : σ), LE.le (f a) N)
      ⊢ Finite ↑(setOf fun s => LE.le (s.sum fun x e => e) N)
    -/
    rw [Set.finite_coe_iff] at this ⊢
    exact this.subset fun n hn i ↦ (eq_or_ne (n i) 0).elim
      (fun h ↦ h.trans_le N.zero_le) fun h ↦
        (Finset.single_le_sum (fun _ _ ↦ Nat.zero_le _) <| Finsupp.mem_support_iff.mpr h).trans hn
  Module.Finite.of_basis (basisRestrictSupport R _)


/--
If `S` is an `R`-algebra, then `MvPolynomial σ S` is a `MvPolynomial σ R` algebra.

Warning: This produces a diamond for
`Algebra (MvPolynomial σ R) (MvPolynomial σ (MvPolynomial σ S))`. That's why it is not a
global instance.
-/
noncomputable def algebraMvPolynomial : Algebra (MvPolynomial σ R) (MvPolynomial σ S) :=
  (MvPolynomial.map (algebraMap R S)).toAlgebra


@[simp]
lemma algebraMap_def :
    algebraMap (MvPolynomial σ R) (MvPolynomial σ S) = MvPolynomial.map (algebraMap R S) :=
  rfl


instance : IsScalarTower R (MvPolynomial σ R) (MvPolynomial σ S) :=
                                      /-
                                        σ✝ : Type u
                                        R✝ : Type v
                                        inst✝³ : CommSemiring R✝
                                        p m : Nat
                                        R : Type u_1
                                        S : Type u_2
                                        σ : Type u_3
                                        inst✝² : CommSemiring R
                                        inst✝¹ : CommSemiring S
                                        inst✝ : Algebra R S
                                        ⊢ Eq (algebraMap R (MvPolynomial σ S)) ((algebraMap (MvPolynomial σ R) (MvPoly …
                                      -/
  IsScalarTower.of_algebraMap_eq' (by ext; simp)
                                           /-
                                             🎉 no goals
                                           -/


