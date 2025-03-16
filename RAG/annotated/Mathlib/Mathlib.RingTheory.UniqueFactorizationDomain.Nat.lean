instance instWfDvdMonoid : WfDvdMonoid ℕ where
  wf := by
    refine RelHomClass.wellFounded
      (⟨fun x : ℕ => if x = 0 then (⊤ : ℕ∞) else x, ?_⟩ : DvdNotUnit →r (· < ·)) wellFounded_lt
    /-
      ⊢ ∀ {a b : Nat}, DvdNotUnit a b → LT.lt ((fun x => ite (Eq x 0) Top.top ↑x) a) …
    -/
    intro a b h
    /-
      a b : Nat
      h : DvdNotUnit a b
      ⊢ LT.lt ((fun x => ite (Eq x 0) Top.top ↑x) a) ((fun x => ite (Eq x 0) Top.top …
    -/
    cases' a with a
      /-
        case zero
        b : Nat
        h : DvdNotUnit 0 b
        ⊢ LT.lt ((fun x => ite (Eq x 0) Top.top ↑x) 0) ((fun x => ite (Eq x 0) Top.top …
      -/
    · exfalso
      /-
        case zero
        b : Nat
        h : DvdNotUnit 0 b
        ⊢ False
      -/
      revert h
      /-
        case zero
        b : Nat
        ⊢ DvdNotUnit 0 b → False
      -/
      simp [DvdNotUnit]
      /-
        🎉 no goals
      -/
    /-
      case succ
      b a : Nat
      h : DvdNotUnit (HAdd.hAdd a 1) b
      ⊢ LT.lt ((fun x => ite (Eq x 0) Top.top ↑x) (HAdd.hAdd a 1)) ((fun x => ite (E …
    -/
    cases b
      /-
        case succ.zero
        a : Nat
        h : DvdNotUnit (HAdd.hAdd a 1) 0
        ⊢ LT.lt ((fun x => ite (Eq x 0) Top.top ↑x) (HAdd.hAdd a 1)) ((fun x => ite (E …
      -/
    · simpa [succ_ne_zero] using ENat.coe_lt_top (a + 1)
      /-
        🎉 no goals
      -/
    /-
      case succ.succ
      a n✝ : Nat
      h : DvdNotUnit (HAdd.hAdd a 1) (HAdd.hAdd n✝ 1)
      ⊢ LT.lt ((fun x => ite (Eq x 0) Top.top ↑x) (HAdd.hAdd a 1)) ((fun x => ite (E …
    -/
    cases' dvd_and_not_dvd_iff.2 h with h1 h2
    /-
      case succ.succ.intro
      a n✝ : Nat
      h : DvdNotUnit (HAdd.hAdd a 1) (HAdd.hAdd n✝ 1)
      h1 : Dvd.dvd (HAdd.hAdd a 1) (HAdd.hAdd n✝ 1)
      h2 : Not (Dvd.dvd (HAdd.hAdd n✝ 1) (HAdd.hAdd a 1))
      ⊢ LT.lt ((fun x => ite (Eq x 0) Top.top ↑x) (HAdd.hAdd a 1)) ((fun x => ite (E …
    -/
    simp only [succ_ne_zero, cast_lt, if_false]
    /-
      case succ.succ.intro
      a n✝ : Nat
      h : DvdNotUnit (HAdd.hAdd a 1) (HAdd.hAdd n✝ 1)
      h1 : Dvd.dvd (HAdd.hAdd a 1) (HAdd.hAdd n✝ 1)
      h2 : Not (Dvd.dvd (HAdd.hAdd n✝ 1) (HAdd.hAdd a 1))
      ⊢ LT.lt (HAdd.hAdd a 1) (HAdd.hAdd n✝ 1)
    -/
    refine lt_of_le_of_ne (Nat.le_of_dvd (Nat.succ_pos _) h1) fun con => h2 ?_
    /-
      case succ.succ.intro
      a n✝ : Nat
      h : DvdNotUnit (HAdd.hAdd a 1) (HAdd.hAdd n✝ 1)
      h1 : Dvd.dvd (HAdd.hAdd a 1) (HAdd.hAdd n✝ 1)
      h2 : Not (Dvd.dvd (HAdd.hAdd n✝ 1) (HAdd.hAdd a 1))
      con : Eq (HAdd.hAdd a 1) (HAdd.hAdd n✝ 1)
      ⊢ Dvd.dvd (HAdd.hAdd n✝ 1) (HAdd.hAdd a 1)
    -/
    rw [con]
    /-
      🎉 no goals
    -/


instance instUniqueFactorizationMonoid : UniqueFactorizationMonoid ℕ where
  irreducible_iff_prime := Nat.irreducible_iff_prime


lemma factors_eq : ∀ n : ℕ, normalizedFactors n = n.primeFactorsList
            /-
              ⊢ Eq (UniqueFactorizationMonoid.normalizedFactors 0) ↑(Nat.primeFactorsList 0)
            -/
  | 0 => by simp
            /-
              🎉 no goals
            -/
  | n + 1 => by
    /-
      n : Nat
      ⊢ Eq (UniqueFactorizationMonoid.normalizedFactors (HAdd.hAdd n 1)) ↑(HAdd.hAdd …
    -/
    rw [← Multiset.rel_eq, ← associated_eq_eq]
    /-
      n : Nat
      ⊢ Multiset.Rel (fun x1 x2 => Associated x1 x2) (UniqueFactorizationMonoid.norm …
    -/
    apply UniqueFactorizationMonoid.factors_unique irreducible_of_normalized_factor _
      /-
        n : Nat
        ⊢ Associated (UniqueFactorizationMonoid.normalizedFactors (HAdd.hAdd n 1)).pro …
      -/
    · rw [Multiset.prod_coe, Nat.prod_primeFactorsList n.succ_ne_zero]
      /-
        n : Nat
        ⊢ Associated (UniqueFactorizationMonoid.normalizedFactors (HAdd.hAdd n 1)).pro …
      -/
      exact prod_normalizedFactors n.succ_ne_zero
      /-
        🎉 no goals
      -/
      /-
        n : Nat
        ⊢ ∀ (x : Nat), Membership.mem (↑(HAdd.hAdd n 1).primeFactorsList) x → Irreduci …
      -/
    · intro x hx
      /-
        n x : Nat
        hx : Membership.mem (↑(HAdd.hAdd n 1).primeFactorsList) x
        ⊢ Irreducible x
      -/
      rw [Nat.irreducible_iff_prime, ← Nat.prime_iff]
      /-
        n x : Nat
        hx : Membership.mem (↑(HAdd.hAdd n 1).primeFactorsList) x
        ⊢ Nat.Prime x
      -/
      exact Nat.prime_of_mem_primeFactorsList hx
      /-
        🎉 no goals
      -/


lemma factors_multiset_prod_of_irreducible {s : Multiset ℕ} (h : ∀ x : ℕ, x ∈ s → Irreducible x) :
    normalizedFactors s.prod = s := by
  /-
    s : Multiset Nat
    h : ∀ (x : Nat), Membership.mem s x → Irreducible x
    ⊢ Eq (UniqueFactorizationMonoid.normalizedFactors s.prod) s
  -/
  rw [← Multiset.rel_eq, ← associated_eq_eq]
  apply UniqueFactorizationMonoid.factors_unique irreducible_of_normalized_factor h
    (prod_normalizedFactors _)
  /-
    s : Multiset Nat
    h : ∀ (x : Nat), Membership.mem s x → Irreducible x
    ⊢ Ne s.prod 0
  -/
  rw [Ne, Multiset.prod_eq_zero_iff]
  /-
    s : Multiset Nat
    h : ∀ (x : Nat), Membership.mem s x → Irreducible x
    ⊢ Not (Membership.mem s 0)
  -/
  exact fun con ↦ not_irreducible_zero (h 0 con)
  /-
    🎉 no goals
  -/


