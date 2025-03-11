/-- The `Finsupp` counterpart of `Multiset.antidiagonal`: the antidiagonal of
`s : α →₀ ℕ` consists of all pairs `(t₁, t₂) : (α →₀ ℕ) × (α →₀ ℕ)` such that `t₁ + t₂ = s`.
The finitely supported function `antidiagonal s` is equal to the multiplicities of these pairs. -/
def antidiagonal' (f : α →₀ ℕ) : (α →₀ ℕ) × (α →₀ ℕ) →₀ ℕ :=
  Multiset.toFinsupp
    ((Finsupp.toMultiset f).antidiagonal.map (Prod.map Multiset.toFinsupp Multiset.toFinsupp))


/-- The antidiagonal of `s : α →₀ ℕ` is the finset of all pairs `(t₁, t₂) : (α →₀ ℕ) × (α →₀ ℕ)`
such that `t₁ + t₂ = s`. -/
instance instHasAntidiagonal : HasAntidiagonal (α →₀ ℕ) where
  antidiagonal f := f.antidiagonal'.support
  mem_antidiagonal {f} {p} := by
    /-
      α : Type u
      inst✝ : DecidableEq α
      f : Finsupp α Nat
      p : Prod (Finsupp α Nat) (Finsupp α Nat)
      ⊢ Iff (Membership.mem ((fun f => f.antidiagonal'.support) f) p) (Eq (HAdd.hAdd …
    -/
    rcases p with ⟨p₁, p₂⟩
    simp [antidiagonal', ← and_assoc, Multiset.toFinsupp_eq_iff,
    ← Multiset.toFinsupp_eq_iff (f := f)]


@[simp]
theorem antidiagonal_zero : antidiagonal (0 : α →₀ ℕ) = singleton (0, 0) := rfl


@[to_additive]
theorem prod_antidiagonal_swap {M : Type*} [CommMonoid M] (n : α →₀ ℕ)
    (f : (α →₀ ℕ) → (α →₀ ℕ) → M) :
    ∏ p ∈ antidiagonal n, f p.1 p.2 = ∏ p ∈ antidiagonal n, f p.2 p.1 :=
                                      /-
                                        α : Type u
                                        inst✝¹ : DecidableEq α
                                        M : Type u_1
                                        inst✝ : CommMonoid M
                                        n : Finsupp α Nat
                                        f : Finsupp α Nat → Finsupp α Nat → M
                                        ⊢ ∀ (i : Prod (Finsupp α Nat) (Finsupp α Nat)), Iff (Membership.mem (Finset.Ha …
                                      -/
                                      /-
                                        🎉 no goals
                                      -/
  prod_equiv (Equiv.prodComm _ _) (by simp [add_comm]) (by simp)
                                                           /-
                                                             🎉 no goals
                                                           -/


@[simp]
theorem antidiagonal_single (a : α) (n : ℕ) :
    antidiagonal (single a n) = (antidiagonal n).map
      (Function.Embedding.prodMap ⟨_, single_injective a⟩ ⟨_, single_injective a⟩) := by
  /-
    α : Type u
    inst✝ : DecidableEq α
    a : α
    n : Nat
    ⊢ Eq (Finset.HasAntidiagonal.antidiagonal (Finsupp.single a n)) (Finset.map ({ …
  -/
  ext ⟨x, y⟩
  simp only [mem_antidiagonal, mem_map, mem_antidiagonal, Function.Embedding.coe_prodMap,
    Function.Embedding.coeFn_mk, Prod.map_apply, Prod.mk.injEq, Prod.exists]
  /-
    case h.mk
    α : Type u
    inst✝ : DecidableEq α
    a : α
    n : Nat
    x y : Finsupp α Nat
    ⊢ Iff (Eq (HAdd.hAdd x y) (Finsupp.single a n)) (Exists fun a_1 => Exists fun  …
  -/
  constructor
    /-
      case h.mk.mp
      α : Type u
      inst✝ : DecidableEq α
      a : α
      n : Nat
      x y : Finsupp α Nat
      ⊢ Eq (HAdd.hAdd x y) (Finsupp.single a n) → Exists fun a_2 => Exists fun b =>  …
    -/
  · intro h
    /-
      case h.mk.mp
      α : Type u
      inst✝ : DecidableEq α
      a : α
      n : Nat
      x y : Finsupp α Nat
      h : Eq (HAdd.hAdd x y) (Finsupp.single a n)
      ⊢ Exists fun a_1 => Exists fun b => And (Eq (HAdd.hAdd a_1 b) n) (And (Eq (Fin …
    -/
    refine ⟨x a, y a, DFunLike.congr_fun h a |>.trans single_eq_same, ?_⟩
    /-
      case h.mk.mp
      α : Type u
      inst✝ : DecidableEq α
      a : α
      n : Nat
      x y : Finsupp α Nat
      h : Eq (HAdd.hAdd x y) (Finsupp.single a n)
      ⊢ And (Eq (Finsupp.single a (x a)) x) (Eq (Finsupp.single a (y a)) y)
    -/
    simp_rw [DFunLike.ext_iff, ← forall_and]
    /-
      case h.mk.mp
      α : Type u
      inst✝ : DecidableEq α
      a : α
      n : Nat
      x y : Finsupp α Nat
      h : Eq (HAdd.hAdd x y) (Finsupp.single a n)
      ⊢ ∀ (x_1 : α), And (Eq ((Finsupp.single a (x a)) x_1) (x x_1)) (Eq ((Finsupp.s …
    -/
    intro i
    /-
      case h.mk.mp
      α : Type u
      inst✝ : DecidableEq α
      a : α
      n : Nat
      x y : Finsupp α Nat
      h : Eq (HAdd.hAdd x y) (Finsupp.single a n)
      i : α
      ⊢ And (Eq ((Finsupp.single a (x a)) i) (x i)) (Eq ((Finsupp.single a (y a)) i) …
    -/
    replace h := DFunLike.congr_fun h i
    /-
      case h.mk.mp
      α : Type u
      inst✝ : DecidableEq α
      a : α
      n : Nat
      x y : Finsupp α Nat
      i : α
      h : Eq ((HAdd.hAdd x y) i) ((Finsupp.single a n) i)
      ⊢ And (Eq ((Finsupp.single a (x a)) i) (x i)) (Eq ((Finsupp.single a (y a)) i) …
    -/
    simp_rw [single_apply, Finsupp.add_apply] at h ⊢
    /-
      case h.mk.mp
      α : Type u
      inst✝ : DecidableEq α
      a : α
      n : Nat
      x y : Finsupp α Nat
      i : α
      h : Eq (HAdd.hAdd (x i) (y i)) (ite (Eq a i) n 0)
      ⊢ And (Eq (ite (Eq a i) (x a) 0) (x i)) (Eq (ite (Eq a i) (y a) 0) (y i))
    -/
    obtain rfl | hai := Decidable.eq_or_ne a i
      /-
        case h.mk.mp.inl
        α : Type u
        inst✝ : DecidableEq α
        a : α
        n : Nat
        x y : Finsupp α Nat
        h : Eq (HAdd.hAdd (x a) (y a)) (ite (Eq a a) n 0)
        ⊢ And (Eq (ite (Eq a a) (x a) 0) (x a)) (Eq (ite (Eq a a) (y a) 0) (y a))
      -/
    · exact ⟨if_pos rfl, if_pos rfl⟩
      /-
        🎉 no goals
      -/
      /-
        case h.mk.mp.inr
        α : Type u
        inst✝ : DecidableEq α
        a : α
        n : Nat
        x y : Finsupp α Nat
        i : α
        h : Eq (HAdd.hAdd (x i) (y i)) (ite (Eq a i) n 0)
        hai : Ne a i
        ⊢ And (Eq (ite (Eq a i) (x a) 0) (x i)) (Eq (ite (Eq a i) (y a) 0) (y i))
      -/
    · simp_rw [if_neg hai, add_eq_zero] at h ⊢
      /-
        case h.mk.mp.inr
        α : Type u
        inst✝ : DecidableEq α
        a : α
        n : Nat
        x y : Finsupp α Nat
        i : α
        hai : Ne a i
        h : And (Eq (x i) 0) (Eq (y i) 0)
        ⊢ And (Eq 0 (x i)) (Eq 0 (y i))
      -/
      exact h.imp Eq.symm Eq.symm
      /-
        🎉 no goals
      -/
    /-
      case h.mk.mpr
      α : Type u
      inst✝ : DecidableEq α
      a : α
      n : Nat
      x y : Finsupp α Nat
      ⊢ (Exists fun a_1 => Exists fun b => And (Eq (HAdd.hAdd a_1 b) n) (And (Eq (Fi …
    -/
  · rintro ⟨a, b, rfl, rfl, rfl⟩
    /-
      case h.mk.mpr.intro.intro.intro.intro
      α : Type u
      inst✝ : DecidableEq α
      a✝ : α
      a b : Nat
      ⊢ Eq (HAdd.hAdd (Finsupp.single a✝ a) (Finsupp.single a✝ b)) (Finsupp.single a …
    -/
    exact (single_add _ _ _).symm
    /-
      🎉 no goals
    -/


