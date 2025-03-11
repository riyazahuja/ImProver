/-- A function `φ : G → G` is fixed-point-free if `1 : G` is the only fixed point of `φ`. -/
def FixedPointFree [One G] := ∀ g, φ g = g → g = 1


/-- The commutator map `g ↦ g / φ g`. If `φ g = h * g * h⁻¹`, then `g / φ g` is exactly the
  commutator `[g, h] = g * h * g⁻¹ * h⁻¹`. -/
def commutatorMap [Div G] (g : G) := g / φ g


@[simp] theorem commutatorMap_apply [Div G] (g : G) : commutatorMap φ g = g / φ g := rfl


theorem commutatorMap_injective (hφ : FixedPointFree φ) : Function.Injective (commutatorMap φ) := by
  /-
    G : Type u_1
    inst✝ : Group G
    φ : MonoidHom G G
    hφ : MonoidHom.FixedPointFree ⇑φ
    ⊢ Function.Injective (MonoidHom.commutatorMap ⇑φ)
  -/
  refine fun x y h ↦ inv_mul_eq_one.mp <| hφ _ ?_
  /-
    G : Type u_1
    inst✝ : Group G
    φ : MonoidHom G G
    hφ : MonoidHom.FixedPointFree ⇑φ
    x y : G
    h : Eq (MonoidHom.commutatorMap (⇑φ) x) (MonoidHom.commutatorMap (⇑φ) y)
    ⊢ Eq (φ (HMul.hMul (Inv.inv x) y)) (HMul.hMul (Inv.inv x) y)
  -/
  rwa [map_mul, map_inv, eq_inv_mul_iff_mul_eq, ← mul_assoc, ← eq_div_iff_mul_eq', ← division_def]
  /-
    🎉 no goals
  -/


theorem commutatorMap_surjective (hφ : FixedPointFree φ) : Function.Surjective (commutatorMap φ) :=
  Finite.surjective_of_injective hφ.commutatorMap_injective


theorem prod_pow_eq_one (hφ : FixedPointFree φ) {n : ℕ} (hn : φ^[n] = _root_.id) (g : G) :
    ((List.range n).map (fun k ↦ φ^[k] g)).prod = 1 := by
  /-
    G : Type u_1
    inst✝¹ : Group G
    φ : MonoidHom G G
    inst✝ : Finite G
    hφ : MonoidHom.FixedPointFree ⇑φ
    n : Nat
    hn : Eq (Nat.iterate (⇑φ) n) _root_.id
    g : G
    ⊢ Eq (List.map (fun k => Nat.iterate (⇑φ) k g) (List.range n)).prod 1
  -/
  obtain ⟨g, rfl⟩ := commutatorMap_surjective hφ g
  /-
    case intro
    G : Type u_1
    inst✝¹ : Group G
    φ : MonoidHom G G
    inst✝ : Finite G
    hφ : MonoidHom.FixedPointFree ⇑φ
    n : Nat
    hn : Eq (Nat.iterate (⇑φ) n) _root_.id
    g : G
    ⊢ Eq (List.map (fun k => Nat.iterate (⇑φ) k (MonoidHom.commutatorMap (⇑φ) g))  …
  -/
  simp only [commutatorMap_apply, iterate_map_div, ← Function.iterate_succ_apply]
  /-
    case intro
    G : Type u_1
    inst✝¹ : Group G
    φ : MonoidHom G G
    inst✝ : Finite G
    hφ : MonoidHom.FixedPointFree ⇑φ
    n : Nat
    hn : Eq (Nat.iterate (⇑φ) n) _root_.id
    g : G
    ⊢ Eq (List.map (fun k => HDiv.hDiv (Nat.iterate (⇑φ) k g) (Nat.iterate (⇑φ) k. …
  -/
  rw [List.prod_range_div', Function.iterate_zero_apply, hn, Function.id_def, div_self']
  /-
    🎉 no goals
  -/


theorem coe_eq_inv_of_sq_eq_one (hφ : FixedPointFree φ) (h2 : φ^[2] = _root_.id) : ⇑φ = (·⁻¹) := by
  /-
    G : Type u_1
    inst✝¹ : Group G
    φ : MonoidHom G G
    inst✝ : Finite G
    hφ : MonoidHom.FixedPointFree ⇑φ
    h2 : Eq (Nat.iterate (⇑φ) 2) _root_.id
    ⊢ Eq ⇑φ fun x => Inv.inv x
  -/
  ext g
  /-
    case h
    G : Type u_1
    inst✝¹ : Group G
    φ : MonoidHom G G
    inst✝ : Finite G
    hφ : MonoidHom.FixedPointFree ⇑φ
    h2 : Eq (Nat.iterate (⇑φ) 2) _root_.id
    g : G
    ⊢ Eq (φ g) (Inv.inv g)
  -/
  have key : g * φ g = 1 := by simpa [List.range_succ] using hφ.prod_pow_eq_one h2 g
  /-
    case h
    G : Type u_1
    inst✝¹ : Group G
    φ : MonoidHom G G
    inst✝ : Finite G
    hφ : MonoidHom.FixedPointFree ⇑φ
    h2 : Eq (Nat.iterate (⇑φ) 2) _root_.id
    g : G
    key : Eq (HMul.hMul g (φ g)) 1
    ⊢ Eq (φ g) (Inv.inv g)
  -/
  rwa [← inv_eq_iff_mul_eq_one, eq_comm] at key
  /-
    🎉 no goals
  -/


theorem coe_eq_inv_of_involutive (hφ : FixedPointFree φ) (h2 : Function.Involutive φ) :
    ⇑φ = (·⁻¹) :=
  coe_eq_inv_of_sq_eq_one hφ  (funext h2)


theorem commute_all_of_involutive (hφ : FixedPointFree φ) (h2 : Function.Involutive φ) (g h : G) :
    Commute g h := by
  /-
    G : Type u_1
    inst✝¹ : Group G
    φ : MonoidHom G G
    inst✝ : Finite G
    hφ : MonoidHom.FixedPointFree ⇑φ
    h2 : Function.Involutive ⇑φ
    g h : G
    ⊢ Commute g h
  -/
  have key := map_mul φ g h
  /-
    G : Type u_1
    inst✝¹ : Group G
    φ : MonoidHom G G
    inst✝ : Finite G
    hφ : MonoidHom.FixedPointFree ⇑φ
    h2 : Function.Involutive ⇑φ
    g h : G
    key : Eq (φ (HMul.hMul g h)) (HMul.hMul (φ g) (φ h))
    ⊢ Commute g h
  -/
  rwa [hφ.coe_eq_inv_of_involutive h2, inv_eq_iff_eq_inv, mul_inv_rev, inv_inv, inv_inv] at key
  /-
    🎉 no goals
  -/


/-- If a finite group admits a fixed-point-free involution, then it is commutative. -/
def commGroupOfInvolutive (hφ : FixedPointFree φ) (h2 : Function.Involutive φ):
    CommGroup G := .mk (hφ.commute_all_of_involutive h2)


theorem orderOf_ne_two_of_involutive (hφ : FixedPointFree φ) (h2 : Function.Involutive φ) (g : G) :
    orderOf g ≠ 2 := by
  /-
    G : Type u_1
    inst✝¹ : Group G
    φ : MonoidHom G G
    inst✝ : Finite G
    hφ : MonoidHom.FixedPointFree ⇑φ
    h2 : Function.Involutive ⇑φ
    g : G
    ⊢ Ne (orderOf g) 2
  -/
  intro hg
  have key : φ g = g := by
    rw [hφ.coe_eq_inv_of_involutive h2, inv_eq_iff_mul_eq_one, ← sq, ← hg, pow_orderOf_eq_one]
  /-
    G : Type u_1
    inst✝¹ : Group G
    φ : MonoidHom G G
    inst✝ : Finite G
    hφ : MonoidHom.FixedPointFree ⇑φ
    h2 : Function.Involutive ⇑φ
    g : G
    hg : Eq (orderOf g) 2
    key : Eq (φ g) g
    ⊢ False
  -/
  rw [hφ g key, orderOf_one] at hg
  /-
    G : Type u_1
    inst✝¹ : Group G
    φ : MonoidHom G G
    inst✝ : Finite G
    hφ : MonoidHom.FixedPointFree ⇑φ
    h2 : Function.Involutive ⇑φ
    g : G
    hg : Eq 1 2
    key : Eq (φ g) g
    ⊢ False
  -/
  contradiction
  /-
    🎉 no goals
  -/


theorem odd_card_of_involutive (hφ : FixedPointFree φ) (h2 : Function.Involutive φ) :
    Odd (Nat.card G) := by
  /-
    G : Type u_1
    inst✝¹ : Group G
    φ : MonoidHom G G
    inst✝ : Finite G
    hφ : MonoidHom.FixedPointFree ⇑φ
    h2 : Function.Involutive ⇑φ
    ⊢ Odd (Nat.card G)
  -/
  have := Fintype.ofFinite G
  /-
    G : Type u_1
    inst✝¹ : Group G
    φ : MonoidHom G G
    inst✝ : Finite G
    hφ : MonoidHom.FixedPointFree ⇑φ
    h2 : Function.Involutive ⇑φ
    this : Fintype G
    ⊢ Odd (Nat.card G)
  -/
  by_contra h
  /-
    G : Type u_1
    inst✝¹ : Group G
    φ : MonoidHom G G
    inst✝ : Finite G
    hφ : MonoidHom.FixedPointFree ⇑φ
    h2 : Function.Involutive ⇑φ
    this : Fintype G
    h : Not (Odd (Nat.card G))
    ⊢ False
  -/
  rw [Nat.not_odd_iff_even, even_iff_two_dvd, Nat.card_eq_fintype_card] at h
  /-
    G : Type u_1
    inst✝¹ : Group G
    φ : MonoidHom G G
    inst✝ : Finite G
    hφ : MonoidHom.FixedPointFree ⇑φ
    h2 : Function.Involutive ⇑φ
    this : Fintype G
    h : Dvd.dvd 2 (Fintype.card G)
    ⊢ False
  -/
  obtain ⟨g, hg⟩ := exists_prime_orderOf_dvd_card 2 h
  /-
    case intro
    G : Type u_1
    inst✝¹ : Group G
    φ : MonoidHom G G
    inst✝ : Finite G
    hφ : MonoidHom.FixedPointFree ⇑φ
    h2 : Function.Involutive ⇑φ
    this : Fintype G
    h : Dvd.dvd 2 (Fintype.card G)
    g : G
    hg : Eq (orderOf g) 2
    ⊢ False
  -/
  exact hφ.orderOf_ne_two_of_involutive h2 g hg
  /-
    🎉 no goals
  -/


theorem odd_orderOf_of_involutive (hφ : FixedPointFree φ) (h2 : Function.Involutive φ) (g : G) :
    Odd (orderOf g) :=
  Odd.of_dvd_nat (hφ.odd_card_of_involutive h2) (orderOf_dvd_natCard g)


