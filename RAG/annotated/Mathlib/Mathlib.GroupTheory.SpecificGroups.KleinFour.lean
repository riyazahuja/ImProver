/-- An (additive) Klein four-group is an (additive) group of cardinality four and exponent two. -/
class IsAddKleinFour (G : Type*) [AddGroup G] : Prop where
  card_four : Nat.card G = 4
  exponent_two : AddMonoid.exponent G = 2


/-- A Klein four-group is a group of cardinality four and exponent two. -/
@[to_additive existing IsAddKleinFour]
class IsKleinFour (G : Type*) [Group G] : Prop where
  card_four : Nat.card G = 4
  exponent_two : Monoid.exponent G = 2


instance : IsAddKleinFour (ZMod 2 × ZMod 2) where
                  /-
                    ⊢ Eq (Nat.card (Prod (ZMod 2) (ZMod 2))) 4
                  -/
  card_four := by simp
                  /-
                    🎉 no goals
                  -/
                     /-
                       ⊢ Eq (AddMonoid.exponent (Prod (ZMod 2) (ZMod 2))) 2
                     -/
  exponent_two := by simp [AddMonoid.exponent_prod]
                     /-
                       🎉 no goals
                     -/


instance : IsKleinFour (DihedralGroup 2) where
                  /-
                    ⊢ Eq (Nat.card (DihedralGroup 2)) 4
                  -/
  card_four := by simp only [Nat.card_eq_fintype_card]; rfl
                                                        /-
                                                          🎉 no goals
                                                        -/
                     /-
                       ⊢ Eq (Monoid.exponent (DihedralGroup 2)) 2
                     -/
  exponent_two := by simp [DihedralGroup.exponent]
                     /-
                       🎉 no goals
                     -/


instance {G : Type*} [Group G] [IsKleinFour G] :
    IsAddKleinFour (Additive G) where
                  /-
                    G : Type u_1
                    inst✝¹ : Group G
                    inst✝ : IsKleinFour G
                    ⊢ Eq (Nat.card (Additive G)) 4
                  -/
  card_four := by rw [← IsKleinFour.card_four (G := G)]; congr!
                                                         /-
                                                           🎉 no goals
                                                         -/
                     /-
                       G : Type u_1
                       inst✝¹ : Group G
                       inst✝ : IsKleinFour G
                       ⊢ Eq (AddMonoid.exponent (Additive G)) 2
                     -/
  exponent_two := by simp
                     /-
                       🎉 no goals
                     -/


instance {G : Type*} [AddGroup G] [IsAddKleinFour G] :
    IsKleinFour (Multiplicative G) where
                  /-
                    G : Type u_1
                    inst✝¹ : AddGroup G
                    inst✝ : IsAddKleinFour G
                    ⊢ Eq (Nat.card (Multiplicative G)) 4
                  -/
  card_four := by rw [← IsAddKleinFour.card_four (G := G)]; congr!
                                                            /-
                                                              🎉 no goals
                                                            -/
                     /-
                       G : Type u_1
                       inst✝¹ : AddGroup G
                       inst✝ : IsAddKleinFour G
                       ⊢ Eq (Monoid.exponent (Multiplicative G)) 2
                     -/
  exponent_two := by simp
                     /-
                       🎉 no goals
                     -/


@[to_additive]
instance instFinite {G : Type*} [Group G] [IsKleinFour G] : Finite G :=
                                   /-
                                     G : Type u_1
                                     inst✝¹ : Group G
                                     inst✝ : IsKleinFour G
                                     ⊢ Ne (Nat.card G) 0
                                   -/
  Nat.finite_of_card_ne_zero <| by norm_num [IsKleinFour.card_four]
                                   /-
                                     🎉 no goals
                                   -/


@[to_additive (attr := simp)]
lemma card_four' {G : Type*} [Group G] [Fintype G] [IsKleinFour G] :
    Fintype.card G = 4 :=
  Nat.card_eq_fintype_card (α := G).symm ▸ IsKleinFour.card_four


@[to_additive]
lemma not_isCyclic : ¬ IsCyclic G :=
             /-
               G : Type u_1
               inst✝¹ : Group G
               inst✝ : IsKleinFour G
               h : IsCyclic G
               ⊢ False
             -/
  fun h ↦ by let _inst := Fintype.ofFinite G; simpa using h.exponent_eq_card
                                              /-
                                                🎉 no goals
                                              -/


@[to_additive]
                                                                       /-
                                                                         G : Type u_1
                                                                         inst✝¹ : Group G
                                                                         inst✝ : IsKleinFour G
                                                                         x : G
                                                                         ⊢ Eq (Monoid.exponent G) 2
                                                                       -/
lemma inv_eq_self (x : G) : x⁻¹ = x := inv_eq_self_of_exponent_two (by simp) x
                                                                       /-
                                                                         🎉 no goals
                                                                       -/

/- this is not an appropriate global `simp` lemma for a `Prop`-mixin class. Indeed, if it were
then every time Lean sees `·⁻¹` it would try to apply `inv_eq_self` which would trigger
type class inference to try and synthesize an `IsKleinFour` instance. -/

@[to_additive]
lemma mul_self (x : G) : x * x = 1 := by
  /-
    G : Type u_1
    inst✝¹ : Group G
    inst✝ : IsKleinFour G
    x : G
    ⊢ Eq (HMul.hMul x x) 1
  -/
  rw [mul_eq_one_iff_eq_inv, inv_eq_self]
  /-
    🎉 no goals
  -/


@[to_additive]
lemma eq_finset_univ [Fintype G] [DecidableEq G]
    {x y : G} (hx : x ≠ 1) (hy : y ≠ 1) (hxy : x ≠ y) : {x * y, x, y, (1 : G)} = Finset.univ := by
  /-
    G : Type u_1
    inst✝³ : Group G
    inst✝² : IsKleinFour G
    inst✝¹ : Fintype G
    inst✝ : DecidableEq G
    x y : G
    hx : Ne x 1
    hy : Ne y 1
    hxy : Ne x y
    ⊢ Eq (Insert.insert (HMul.hMul x y) (Insert.insert x (Insert.insert y (Singlet …
  -/
  apply Finset.eq_univ_of_card
  /-
    case hs
    G : Type u_1
    inst✝³ : Group G
    inst✝² : IsKleinFour G
    inst✝¹ : Fintype G
    inst✝ : DecidableEq G
    x y : G
    hx : Ne x 1
    hy : Ne y 1
    hxy : Ne x y
    ⊢ Eq (Insert.insert (HMul.hMul x y) (Insert.insert x (Insert.insert y (Singlet …
  -/
  rw [card_four']
  /-
    case hs
    G : Type u_1
    inst✝³ : Group G
    inst✝² : IsKleinFour G
    inst✝¹ : Fintype G
    inst✝ : DecidableEq G
    x y : G
    hx : Ne x 1
    hy : Ne y 1
    hxy : Ne x y
    ⊢ Eq (Insert.insert (HMul.hMul x y) (Insert.insert x (Insert.insert y (Singlet …
  -/
  repeat rw [card_insert_of_not_mem]
  /-
    case hs
    G : Type u_1
    inst✝³ : Group G
    inst✝² : IsKleinFour G
    inst✝¹ : Fintype G
    inst✝ : DecidableEq G
    x y : G
    hx : Ne x 1
    hy : Ne y 1
    hxy : Ne x y
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (Singleton.singleton 1).card 1) 1) 1) 4
  -/
  on_goal 4 => simpa using mul_not_mem_of_exponent_two (by simp) hx hy hxy
  /-
    case hs
    G : Type u_1
    inst✝³ : Group G
    inst✝² : IsKleinFour G
    inst✝¹ : Fintype G
    inst✝ : DecidableEq G
    x y : G
    hx : Ne x 1
    hy : Ne y 1
    hxy : Ne x y
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (Singleton.singleton 1).card 1) 1) 1) 4
  -/
  all_goals aesop
  /-
    🎉 no goals
  -/


@[to_additive]
lemma eq_mul_of_ne_all {x y z : G} (hx : x ≠ 1)
    (hy : y ≠ 1) (hxy : x ≠ y) (hz : z ≠ 1) (hzx : z ≠ x) (hzy : z ≠ y) : z = x * y := by
  classical
  let _ := Fintype.ofFinite G
  apply eq_of_mem_insert_of_not_mem <| (eq_finset_univ hx hy hxy).symm ▸ mem_univ _
  simpa only [mem_singleton, mem_insert, not_or] using ⟨hzx, hzy, hz⟩


/-- An equivalence between an `IsKleinFour` group `G₁` and a group `G₂` of exponent two which sends
`1 : G₁` to `1 : G₂` is in fact an isomorphism. -/
@[to_additive "An equivalence between an `IsAddKleinFour` group `G₁` and a group `G₂` of exponent
two which sends `0 : G₁` to `0 : G₂` is in fact an isomorphism."]
def mulEquiv' (e : G₁ ≃ G₂) (he : e 1 = 1) (h : Monoid.exponent G₂ = 2) : G₁ ≃* G₂ where
  toEquiv := e
  map_mul' := by
    /-
      G : Type u_1
      inst✝⁴ : Group G
      inst✝³ : IsKleinFour G
      G₁ : Type u_2
      G₂ : Type u_3
      inst✝² : Group G₁
      inst✝¹ : Group G₂
      inst✝ : IsKleinFour G₁
      e : Equiv G₁ G₂
      he : Eq (e 1) 1
      h : Eq (Monoid.exponent G₂) 2
      ⊢ ∀ (x y : G₁), Eq (e.toFun (HMul.hMul x y)) (HMul.hMul (e.toFun x) (e.toFun y))
    -/
    let _inst₁ := Fintype.ofFinite G₁
    /-
      G : Type u_1
      inst✝⁴ : Group G
      inst✝³ : IsKleinFour G
      G₁ : Type u_2
      G₂ : Type u_3
      inst✝² : Group G₁
      inst✝¹ : Group G₂
      inst✝ : IsKleinFour G₁
      e : Equiv G₁ G₂
      he : Eq (e 1) 1
      h : Eq (Monoid.exponent G₂) 2
      _inst₁ : Fintype G₁ := Fintype.ofFinite G₁
      ⊢ ∀ (x y : G₁), Eq (e.toFun (HMul.hMul x y)) (HMul.hMul (e.toFun x) (e.toFun y))
    -/
    let _inst₂ := Fintype.ofEquiv G₁ e
    /-
      G : Type u_1
      inst✝⁴ : Group G
      inst✝³ : IsKleinFour G
      G₁ : Type u_2
      G₂ : Type u_3
      inst✝² : Group G₁
      inst✝¹ : Group G₂
      inst✝ : IsKleinFour G₁
      e : Equiv G₁ G₂
      he : Eq (e 1) 1
      h : Eq (Monoid.exponent G₂) 2
      _inst₁ : Fintype G₁ := Fintype.ofFinite G₁
      _inst₂ : Fintype G₂ := Fintype.ofEquiv G₁ e
      ⊢ ∀ (x y : G₁), Eq (e.toFun (HMul.hMul x y)) (HMul.hMul (e.toFun x) (e.toFun y))
    -/
    intro x y
    /-
      G : Type u_1
      inst✝⁴ : Group G
      inst✝³ : IsKleinFour G
      G₁ : Type u_2
      G₂ : Type u_3
      inst✝² : Group G₁
      inst✝¹ : Group G₂
      inst✝ : IsKleinFour G₁
      e : Equiv G₁ G₂
      he : Eq (e 1) 1
      h : Eq (Monoid.exponent G₂) 2
      _inst₁ : Fintype G₁ := Fintype.ofFinite G₁
      _inst₂ : Fintype G₂ := Fintype.ofEquiv G₁ e
      x y : G₁
      ⊢ Eq (e.toFun (HMul.hMul x y)) (HMul.hMul (e.toFun x) (e.toFun y))
    -/
    by_cases hx : x = 1 <;> by_cases hy : y = 1
    /-
      case pos
      G : Type u_1
      inst✝⁴ : Group G
      inst✝³ : IsKleinFour G
      G₁ : Type u_2
      G₂ : Type u_3
      inst✝² : Group G₁
      inst✝¹ : Group G₂
      inst✝ : IsKleinFour G₁
      e : Equiv G₁ G₂
      he : Eq (e 1) 1
      h : Eq (Monoid.exponent G₂) 2
      _inst₁ : Fintype G₁ := Fintype.ofFinite G₁
      _inst₂ : Fintype G₂ := Fintype.ofEquiv G₁ e
      x y : G₁
      hx : Eq x 1
      hy : Eq y 1
      ⊢ Eq (e.toFun (HMul.hMul x y)) (HMul.hMul (e.toFun x) (e.toFun y))
    -/
    all_goals try simp only [hx, hy, mul_one, one_mul, Equiv.toFun_as_coe, he]
    /-
      case neg
      G : Type u_1
      inst✝⁴ : Group G
      inst✝³ : IsKleinFour G
      G₁ : Type u_2
      G₂ : Type u_3
      inst✝² : Group G₁
      inst✝¹ : Group G₂
      inst✝ : IsKleinFour G₁
      e : Equiv G₁ G₂
      he : Eq (e 1) 1
      h : Eq (Monoid.exponent G₂) 2
      _inst₁ : Fintype G₁ := Fintype.ofFinite G₁
      _inst₂ : Fintype G₂ := Fintype.ofEquiv G₁ e
      x y : G₁
      hx : Not (Eq x 1)
      hy : Not (Eq y 1)
      ⊢ Eq (e (HMul.hMul x y)) (HMul.hMul (e x) (e y))
    -/
    by_cases hxy : x = y
      /-
        case pos
        G : Type u_1
        inst✝⁴ : Group G
        inst✝³ : IsKleinFour G
        G₁ : Type u_2
        G₂ : Type u_3
        inst✝² : Group G₁
        inst✝¹ : Group G₂
        inst✝ : IsKleinFour G₁
        e : Equiv G₁ G₂
        he : Eq (e 1) 1
        h : Eq (Monoid.exponent G₂) 2
        _inst₁ : Fintype G₁ := Fintype.ofFinite G₁
        _inst₂ : Fintype G₂ := Fintype.ofEquiv G₁ e
        x y : G₁
        hx : Not (Eq x 1)
        hy : Not (Eq y 1)
        hxy : Eq x y
        ⊢ Eq (e (HMul.hMul x y)) (HMul.hMul (e x) (e y))
      -/
    · simp [hxy, mul_self, ← pow_two (e y), h ▸ Monoid.pow_exponent_eq_one (e y), he]
      /-
        🎉 no goals
      -/
    · classical
      have univ₂ : {e (x * y), e x, e y, (1 : G₂)} = Finset.univ := by
        simpa [map_univ_equiv e, map_insert, he]
          using congr(Finset.map e.toEmbedding $(eq_finset_univ hx hy hxy))
      rw [← Ne, ← e.injective.ne_iff] at hx hy hxy
      rw [he] at hx hy
      symm
      apply eq_of_mem_insert_of_not_mem <| univ₂.symm ▸ mem_univ _
      simpa using mul_not_mem_of_exponent_two h hx hy hxy


/-- Any two `IsKleinFour` groups are isomorphic via any equivalence which sends the identity of one
group to the identity of the other. -/
@[to_additive "Any two `IsAddKleinFour` groups are isomorphic via any
equivalence which sends the identity of one group to the identity of the other."]
abbrev mulEquiv [IsKleinFour G₂] (e : G₁ ≃ G₂) (he : e 1 = 1) : G₁ ≃* G₂ :=
  mulEquiv' e he exponent_two


/-- Any two `IsKleinFour` groups are isomorphic. -/
@[to_additive "Any two `IsAddKleinFour` groups are isomorphic."]
lemma nonempty_mulEquiv [IsKleinFour G₂] : Nonempty (G₁ ≃* G₂) := by
  classical
  let _inst₁ := Fintype.ofFinite G₁
  let _inst₁ := Fintype.ofFinite G₂
  exact ⟨mulEquiv ((Fintype.equivOfCardEq <| by simp).setValue 1 1) <| by simp⟩


