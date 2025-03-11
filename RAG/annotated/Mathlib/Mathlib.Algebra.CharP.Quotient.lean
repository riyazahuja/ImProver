theorem quotient (R : Type u) [CommRing R] (p : ℕ) [hp1 : Fact p.Prime] (hp2 : ↑p ∈ nonunits R) :
    CharP (R ⧸ (Ideal.span ({(p : R)} : Set R) : Ideal R)) p :=
  have hp0 : (p : R ⧸ (Ideal.span {(p : R)} : Ideal R)) = 0 :=
    map_natCast (Ideal.Quotient.mk (Ideal.span {(p : R)} : Ideal R)) p ▸
      Ideal.Quotient.eq_zero_iff_mem.2 (Ideal.subset_span <| Set.mem_singleton _)
  ringChar.of_eq <|
    Or.resolve_left ((Nat.dvd_prime hp1.1).1 <| ringChar.dvd hp0) fun h1 =>
      hp2 <|
        isUnit_iff_dvd_one.2 <|
          Ideal.mem_span_singleton.1 <|
            Ideal.Quotient.eq_zero_iff_mem.1 <|
              @Subsingleton.elim _ (@CharOne.subsingleton _ _ (ringChar.of_eq h1)) _ _


/-- If an ideal does not contain any coercions of natural numbers other than zero, then its quotient
inherits the characteristic of the underlying ring. -/
theorem quotient' {R : Type*} [CommRing R] (p : ℕ) [CharP R p] (I : Ideal R)
    (h : ∀ x : ℕ, (x : R) ∈ I → (x : R) = 0) : CharP (R ⧸ I) p :=
  ⟨fun x => by
    /-
      R : Type u_1
      inst✝¹ : CommRing R
      p : Nat
      inst✝ : CharP R p
      I : Ideal R
      h : ∀ (x : Nat), Membership.mem I ↑x → Eq (↑x) 0
      x : Nat
      ⊢ Iff (Eq (↑x) 0) (Dvd.dvd p x)
    -/
    rw [← cast_eq_zero_iff R p x, ← map_natCast (Ideal.Quotient.mk I)]
    /-
      R : Type u_1
      inst✝¹ : CommRing R
      p : Nat
      inst✝ : CharP R p
      I : Ideal R
      h : ∀ (x : Nat), Membership.mem I ↑x → Eq (↑x) 0
      x : Nat
      ⊢ Iff (Eq ((Ideal.Quotient.mk I) ↑x) 0) (Eq (↑x) 0)
    -/
    refine Ideal.Quotient.eq.trans (?_ : ↑x - 0 ∈ I ↔ _)
    /-
      R : Type u_1
      inst✝¹ : CommRing R
      p : Nat
      inst✝ : CharP R p
      I : Ideal R
      h : ∀ (x : Nat), Membership.mem I ↑x → Eq (↑x) 0
      x : Nat
      ⊢ Iff (Membership.mem I (HSub.hSub (↑x) 0)) (Eq (↑x) 0)
    -/
    rw [sub_zero]
    /-
      R : Type u_1
      inst✝¹ : CommRing R
      p : Nat
      inst✝ : CharP R p
      I : Ideal R
      h : ∀ (x : Nat), Membership.mem I ↑x → Eq (↑x) 0
      x : Nat
      ⊢ Iff (Membership.mem I ↑x) (Eq (↑x) 0)
    -/
    exact ⟨h x, fun h' => h'.symm ▸ I.zero_mem⟩⟩
    /-
      🎉 no goals
    -/


/-- `CharP.quotient'` as an `Iff`. -/
theorem quotient_iff {R : Type*} [CommRing R] (n : ℕ) [CharP R n] (I : Ideal R) :
    CharP (R ⧸ I) n ↔ ∀ x : ℕ, ↑x ∈ I → (x : R) = 0 := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    n : Nat
    inst✝ : CharP R n
    I : Ideal R
    ⊢ Iff (CharP (HasQuotient.Quotient R I) n) (∀ (x : Nat), Membership.mem I ↑x → …
  -/
  refine ⟨fun _ x hx => ?_, CharP.quotient' n I⟩
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    n : Nat
    inst✝ : CharP R n
    I : Ideal R
    x✝ : CharP (HasQuotient.Quotient R I) n
    x : Nat
    hx : Membership.mem I ↑x
    ⊢ Eq (↑x) 0
  -/
  rw [CharP.cast_eq_zero_iff R n, ← CharP.cast_eq_zero_iff (R ⧸ I) n _]
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    n : Nat
    inst✝ : CharP R n
    I : Ideal R
    x✝ : CharP (HasQuotient.Quotient R I) n
    x : Nat
    hx : Membership.mem I ↑x
    ⊢ Eq (↑x) 0
  -/
  exact (Submodule.Quotient.mk_eq_zero I).mpr hx
  /-
    🎉 no goals
  -/


/-- `CharP.quotient_iff`, but stated in terms of inclusions of ideals. -/
theorem quotient_iff_le_ker_natCast {R : Type*} [CommRing R] (n : ℕ) [CharP R n] (I : Ideal R) :
    CharP (R ⧸ I) n ↔ I.comap (Nat.castRingHom R) ≤ RingHom.ker (Nat.castRingHom R) := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    n : Nat
    inst✝ : CharP R n
    I : Ideal R
    ⊢ Iff (CharP (HasQuotient.Quotient R I) n) (LE.le (Ideal.comap (Nat.castRingHo …
  -/
  rw [CharP.quotient_iff, RingHom.ker_eq_comap_bot]; rfl
                                                     /-
                                                       🎉 no goals
                                                     -/


theorem Ideal.Quotient.index_eq_zero {R : Type*} [CommRing R] (I : Ideal R) :
    (↑I.toAddSubgroup.index : R ⧸ I) = 0 := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    I : Ideal R
    ⊢ Eq (↑(Submodule.toAddSubgroup I).index) 0
  -/
  rw [AddSubgroup.index, Nat.card_eq]
  /-
    R : Type u_1
    inst✝ : CommRing R
    I : Ideal R
    ⊢ Eq (↑(dite (Finite (HasQuotient.Quotient R (Submodule.toAddSubgroup I))) (fu …
  -/
  split_ifs with hq; swap
    /-
      case neg
      R : Type u_1
      inst✝ : CommRing R
      I : Ideal R
      hq : Not (Finite (HasQuotient.Quotient R (Submodule.toAddSubgroup I)))
      ⊢ Eq (↑0) 0
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case pos
    R : Type u_1
    inst✝ : CommRing R
    I : Ideal R
    hq : Finite (HasQuotient.Quotient R (Submodule.toAddSubgroup I))
    ⊢ Eq (↑(Fintype.card (HasQuotient.Quotient R (Submodule.toAddSubgroup I)))) 0
  -/
  letI : Fintype (R ⧸ I) := @Fintype.ofFinite _ hq
  /-
    case pos
    R : Type u_1
    inst✝ : CommRing R
    I : Ideal R
    hq : Finite (HasQuotient.Quotient R (Submodule.toAddSubgroup I))
    this : Fintype (HasQuotient.Quotient R I) := Fintype.ofFinite (HasQuotient.Quo …
    ⊢ Eq (↑(Fintype.card (HasQuotient.Quotient R (Submodule.toAddSubgroup I)))) 0
  -/
  exact Nat.cast_card_eq_zero (R ⧸ I)
  /-
    🎉 no goals
  -/

