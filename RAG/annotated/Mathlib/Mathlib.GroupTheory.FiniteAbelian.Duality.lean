private
lemma dvd_exponent {ι G : Type*} [Finite ι] [CommGroup G] {n : ι → ℕ}
    (e : G ≃* ((i : ι) → Multiplicative (ZMod (n i)))) (i : ι) :
    n i ∣ Monoid.exponent G := by
  classical -- to get `DecidableEq ι`
  have : n i = orderOf (e.symm <| Pi.mulSingle i <| .ofAdd 1) := by
    simpa only [MulEquiv.orderOf_eq, orderOf_piMulSingle, orderOf_ofAdd_eq_addOrderOf]
      using (ZMod.addOrderOf_one (n i)).symm
  exact this ▸ Monoid.order_dvd_exponent _


private
lemma exists_apply_ne_one_aux
    (H : ∀ n : ℕ, n ∣ Monoid.exponent G → ∀ a : ZMod n, a ≠ 0 →
      ∃ φ : Multiplicative (ZMod n) →* M, φ (.ofAdd a) ≠ 1)
    {a : G} (ha : a ≠ 1) :
    ∃ φ : G →* M, φ a ≠ 1 := by
  /-
    G : Type u_1
    M : Type u_2
    inst✝² : CommGroup G
    inst✝¹ : Finite G
    inst✝ : CommMonoid M
    H : ∀ (n : Nat), Dvd.dvd n (Monoid.exponent G) → ∀ (a : ZMod n), Ne a 0 → Exis …
    a : G
    ha : Ne a 1
    ⊢ Exists fun φ => Ne (φ a) 1
  -/
  obtain ⟨ι, _, n, _, h⟩ := CommGroup.equiv_prod_multiplicative_zmod_of_finite G
  /-
    case intro.intro.intro.intro
    G : Type u_1
    M : Type u_2
    inst✝² : CommGroup G
    inst✝¹ : Finite G
    inst✝ : CommMonoid M
    H : ∀ (n : Nat), Dvd.dvd n (Monoid.exponent G) → ∀ (a : ZMod n), Ne a 0 → Exis …
    a : G
    ha : Ne a 1
    ι : Type
    w✝ : Fintype ι
    n : ι → Nat
    left✝ : ∀ (i : ι), LT.lt 1 (n i)
    h : Nonempty (MulEquiv G ((i : ι) → Multiplicative (ZMod (n i))))
    ⊢ Exists fun φ => Ne (φ a) 1
  -/
  let e := h.some
  obtain ⟨i, hi⟩ : ∃ i : ι, e a i ≠ 1 := by
    contrapose! ha
    exact (MulEquiv.map_eq_one_iff e).mp <| funext ha
  have hi : (e a i).toAdd ≠ 0 := by
    simp only [ne_eq, toAdd_eq_zero, hi, not_false_eq_true]
  /-
    case intro.intro.intro.intro.intro
    G : Type u_1
    M : Type u_2
    inst✝² : CommGroup G
    inst✝¹ : Finite G
    inst✝ : CommMonoid M
    H : ∀ (n : Nat), Dvd.dvd n (Monoid.exponent G) → ∀ (a : ZMod n), Ne a 0 → Exis …
    a : G
    ha : Ne a 1
    ι : Type
    w✝ : Fintype ι
    n : ι → Nat
    left✝ : ∀ (i : ι), LT.lt 1 (n i)
    h : Nonempty (MulEquiv G ((i : ι) → Multiplicative (ZMod (n i))))
    e : MulEquiv G ((i : ι) → Multiplicative (ZMod (n i))) := h.some
    i : ι
    hi✝ : Ne (e a i) 1
    hi : Ne (Multiplicative.toAdd (e a i)) 0
    ⊢ Exists fun φ => Ne (φ a) 1
  -/
  obtain ⟨φi, hφi⟩ := H (n i) (dvd_exponent e i) ((e a i).toAdd) hi
  /-
    case intro.intro.intro.intro.intro.intro
    G : Type u_1
    M : Type u_2
    inst✝² : CommGroup G
    inst✝¹ : Finite G
    inst✝ : CommMonoid M
    H : ∀ (n : Nat), Dvd.dvd n (Monoid.exponent G) → ∀ (a : ZMod n), Ne a 0 → Exis …
    a : G
    ha : Ne a 1
    ι : Type
    w✝ : Fintype ι
    n : ι → Nat
    left✝ : ∀ (i : ι), LT.lt 1 (n i)
    h : Nonempty (MulEquiv G ((i : ι) → Multiplicative (ZMod (n i))))
    e : MulEquiv G ((i : ι) → Multiplicative (ZMod (n i))) := h.some
    i : ι
    hi✝ : Ne (e a i) 1
    hi : Ne (Multiplicative.toAdd (e a i)) 0
    φi : MonoidHom (Multiplicative (ZMod (n i))) M
    hφi : Ne (φi (Multiplicative.ofAdd (Multiplicative.toAdd (e a i)))) 1
    ⊢ Exists fun φ => Ne (φ a) 1
  -/
  use (φi.comp (Pi.evalMonoidHom (fun (i : ι) ↦ Multiplicative (ZMod (n i))) i)).comp e
  /-
    case h
    G : Type u_1
    M : Type u_2
    inst✝² : CommGroup G
    inst✝¹ : Finite G
    inst✝ : CommMonoid M
    H : ∀ (n : Nat), Dvd.dvd n (Monoid.exponent G) → ∀ (a : ZMod n), Ne a 0 → Exis …
    a : G
    ha : Ne a 1
    ι : Type
    w✝ : Fintype ι
    n : ι → Nat
    left✝ : ∀ (i : ι), LT.lt 1 (n i)
    h : Nonempty (MulEquiv G ((i : ι) → Multiplicative (ZMod (n i))))
    e : MulEquiv G ((i : ι) → Multiplicative (ZMod (n i))) := h.some
    i : ι
    hi✝ : Ne (e a i) 1
    hi : Ne (Multiplicative.toAdd (e a i)) 0
    φi : MonoidHom (Multiplicative (ZMod (n i))) M
    hφi : Ne (φi (Multiplicative.ofAdd (Multiplicative.toAdd (e a i)))) 1
    ⊢ Ne (((φi.comp (Pi.evalMonoidHom (fun i => Multiplicative (ZMod (n i))) i)).c …
  -/
  simpa only [coe_comp, coe_coe, Function.comp_apply, Pi.evalMonoidHom_apply, ne_eq] using hφi
  /-
    🎉 no goals
  -/


/-- If `G` is a finite commutative group of exponent `n` and `M` is a commutative monoid
with enough `n`th roots of unity, then for each `a ≠ 1` in `G`, there exists a
group homomorphism `φ : G → Mˣ` such that `φ a ≠ 1`. -/
theorem exists_apply_ne_one_of_hasEnoughRootsOfUnity {a : G} (ha : a ≠ 1) :
    ∃ φ : G →* Mˣ, φ a ≠ 1 := by
  /-
    G : Type u_1
    M : Type u_2
    inst✝³ : CommGroup G
    inst✝² : Finite G
    inst✝¹ : CommMonoid M
    inst✝ : HasEnoughRootsOfUnity M (Monoid.exponent G)
    a : G
    ha : Ne a 1
    ⊢ Exists fun φ => Ne (φ a) 1
  -/
  refine exists_apply_ne_one_aux G Mˣ (fun n hn a ha₀ ↦ ?_) ha
  /-
    G : Type u_1
    M : Type u_2
    inst✝³ : CommGroup G
    inst✝² : Finite G
    inst✝¹ : CommMonoid M
    inst✝ : HasEnoughRootsOfUnity M (Monoid.exponent G)
    a✝ : G
    ha : Ne a✝ 1
    n : Nat
    hn : Dvd.dvd n (Monoid.exponent G)
    a : ZMod n
    ha₀ : Ne a 0
    ⊢ Exists fun φ => Ne (φ (Multiplicative.ofAdd a)) 1
  -/
  have : NeZero n := ⟨fun H ↦ NeZero.ne _ <| Nat.eq_zero_of_zero_dvd (H ▸ hn)⟩
  /-
    G : Type u_1
    M : Type u_2
    inst✝³ : CommGroup G
    inst✝² : Finite G
    inst✝¹ : CommMonoid M
    inst✝ : HasEnoughRootsOfUnity M (Monoid.exponent G)
    a✝ : G
    ha : Ne a✝ 1
    n : Nat
    hn : Dvd.dvd n (Monoid.exponent G)
    a : ZMod n
    ha₀ : Ne a 0
    this : NeZero n
    ⊢ Exists fun φ => Ne (φ (Multiplicative.ofAdd a)) 1
  -/
  have := HasEnoughRootsOfUnity.of_dvd M hn
  /-
    G : Type u_1
    M : Type u_2
    inst✝³ : CommGroup G
    inst✝² : Finite G
    inst✝¹ : CommMonoid M
    inst✝ : HasEnoughRootsOfUnity M (Monoid.exponent G)
    a✝ : G
    ha : Ne a✝ 1
    n : Nat
    hn : Dvd.dvd n (Monoid.exponent G)
    a : ZMod n
    ha₀ : Ne a 0
    this✝ : NeZero n
    this : HasEnoughRootsOfUnity M n
    ⊢ Exists fun φ => Ne (φ (Multiplicative.ofAdd a)) 1
  -/
  exact ZMod.exists_monoidHom_apply_ne_one (HasEnoughRootsOfUnity.exists_primitiveRoot M n) ha₀
  /-
    🎉 no goals
  -/


/-- A finite commutative group `G` is (noncanonically) isomorphic to the group `G →* Mˣ`
when `M` is a commutative monoid with enough `n`th roots of unity, where `n` is the exponent
of `G`. -/
theorem monoidHom_mulEquiv_of_hasEnoughRootsOfUnity : Nonempty ((G →* Mˣ) ≃* G) := by
  classical -- to get `DecidableEq ι`
  obtain ⟨ι, _, n, ⟨h₁, h₂⟩⟩ := equiv_prod_multiplicative_zmod_of_finite G
  let e := h₂.some
  let e' := Pi.monoidHomMulEquiv (fun i ↦ Multiplicative (ZMod (n i))) Mˣ
  let e'' := MulEquiv.monoidHomCongr e (.refl Mˣ)
  have : ∀ i, NeZero (n i) := fun i ↦ NeZero.of_gt (h₁ i)
  have inst i : HasEnoughRootsOfUnity M <| Nat.card <| Multiplicative <| ZMod (n i) := by
    have hdvd : Nat.card (Multiplicative (ZMod (n i))) ∣ Monoid.exponent G := by
      simpa only [Nat.card_eq_fintype_card, Fintype.card_multiplicative, ZMod.card]
        using dvd_exponent e i
    exact HasEnoughRootsOfUnity.of_dvd M hdvd
  let E i := (IsCyclic.monoidHom_equiv_self (Multiplicative (ZMod (n i))) M).some
  exact ⟨e''.trans <| e'.trans <| (MulEquiv.piCongrRight E).trans e.symm⟩


