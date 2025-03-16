local infixl:50 " ~ᵤ " => Associated


/-- `FactorSet α` representation elements of unique factorization domain as multisets.
`Multiset α` produced by `normalizedFactors` are only unique up to associated elements, while the
multisets in `FactorSet α` are unique by equality and restricted to irreducible elements. This
gives us a representation of each element as a unique multisets (or the added ⊤ for 0), which has a
complete lattice structure. Infimum is the greatest common divisor and supremum is the least common
multiple.
-/
abbrev FactorSet.{u} (α : Type u) [CancelCommMonoidWithZero α] : Type u :=
  WithTop (Multiset { a : Associates α // Irreducible a })


theorem FactorSet.coe_add {a b : Multiset { a : Associates α // Irreducible a }} :
                                           /-
                                             α : Type u_1
                                             inst✝ : CancelCommMonoidWithZero α
                                             a b : Multiset (Subtype fun a => Irreducible a)
                                             ⊢ Eq (↑(HAdd.hAdd a b)) (HAdd.hAdd ↑a ↑b)
                                           -/
    (↑(a + b) : FactorSet α) = a + b := by norm_cast
                                           /-
                                             🎉 no goals
                                           -/


theorem FactorSet.sup_add_inf_eq_add [DecidableEq (Associates α)] :
    ∀ a b : FactorSet α, a ⊔ b + a ⊓ b = a + b
                                          /-
                                            α : Type u_1
                                            inst✝¹ : CancelCommMonoidWithZero α
                                            inst✝ : DecidableEq (Associates α)
                                            b : Associates.FactorSet α
                                            ⊢ Eq (HAdd.hAdd (Max.max Top.top b) (Min.min Top.top b)) (HAdd.hAdd Top.top b)
                                          -/
  | ⊤, b => show ⊤ ⊔ b + ⊤ ⊓ b = ⊤ + b by simp
                                          /-
                                            🎉 no goals
                                          -/
                                          /-
                                            α : Type u_1
                                            inst✝¹ : CancelCommMonoidWithZero α
                                            inst✝ : DecidableEq (Associates α)
                                            a : Associates.FactorSet α
                                            ⊢ Eq (HAdd.hAdd (Max.max a Top.top) (Min.min a Top.top)) (HAdd.hAdd a Top.top)
                                          -/
  | a, ⊤ => show a ⊔ ⊤ + a ⊓ ⊤ = a + ⊤ by simp
                                          /-
                                            🎉 no goals
                                          -/
  | WithTop.some a, WithTop.some b =>
    show (a : FactorSet α) ⊔ b + (a : FactorSet α) ⊓ b = a + b by
      rw [← WithTop.coe_sup, ← WithTop.coe_inf, ← WithTop.coe_add, ← WithTop.coe_add,
        WithTop.coe_eq_coe]
      /-
        α : Type u_1
        inst✝¹ : CancelCommMonoidWithZero α
        inst✝ : DecidableEq (Associates α)
        a b : Multiset (Subtype fun a => Irreducible a)
        ⊢ Eq (HAdd.hAdd (Max.max a b) (Min.min a b)) (HAdd.hAdd a b)
      -/
      exact Multiset.union_add_inter _ _
      /-
        🎉 no goals
      -/


/-- Evaluates the product of a `FactorSet` to be the product of the corresponding multiset,
  or `0` if there is none. -/
def FactorSet.prod : FactorSet α → Associates α
  | ⊤ => 0
  | WithTop.some s => (s.map (↑)).prod


@[simp]
theorem prod_top : (⊤ : FactorSet α).prod = 0 :=
  rfl


@[simp]
theorem prod_coe {s : Multiset { a : Associates α // Irreducible a }} :
    FactorSet.prod (s : FactorSet α) = (s.map (↑)).prod :=
  rfl


@[simp]
theorem prod_add : ∀ a b : FactorSet α, (a + b).prod = a.prod * b.prod
                                                                   /-
                                                                     α : Type u_1
                                                                     inst✝ : CancelCommMonoidWithZero α
                                                                     b : Associates.FactorSet α
                                                                     ⊢ Eq (HAdd.hAdd Top.top b).prod (HMul.hMul Top.top.prod b.prod)
                                                                   -/
  | ⊤, b => show (⊤ + b).prod = (⊤ : FactorSet α).prod * b.prod by simp
                                                                   /-
                                                                     🎉 no goals
                                                                   -/
                                                                   /-
                                                                     α : Type u_1
                                                                     inst✝ : CancelCommMonoidWithZero α
                                                                     a : Associates.FactorSet α
                                                                     ⊢ Eq (HAdd.hAdd a Top.top).prod (HMul.hMul a.prod Top.top.prod)
                                                                   -/
  | a, ⊤ => show (a + ⊤).prod = a.prod * (⊤ : FactorSet α).prod by simp
                                                                   /-
                                                                     🎉 no goals
                                                                   -/
  | WithTop.some a, WithTop.some b => by
    /-
      α : Type u_1
      inst✝ : CancelCommMonoidWithZero α
      a b : Multiset (Subtype fun a => Irreducible a)
      ⊢ Eq (HAdd.hAdd ↑a ↑b).prod (HMul.hMul (Associates.FactorSet.prod ↑a) (Associa …
    -/
    rw [← FactorSet.coe_add, prod_coe, prod_coe, prod_coe, Multiset.map_add, Multiset.prod_add]
    /-
      🎉 no goals
    -/


@[gcongr]
theorem prod_mono : ∀ {a b : FactorSet α}, a ≤ b → a.prod ≤ b.prod
  | ⊤, b, h => by
    /-
      α : Type u_1
      inst✝ : CancelCommMonoidWithZero α
      b : Associates.FactorSet α
      h : LE.le Top.top b
      ⊢ LE.le Top.top.prod b.prod
    -/
    have : b = ⊤ := top_unique h
    /-
      α : Type u_1
      inst✝ : CancelCommMonoidWithZero α
      b : Associates.FactorSet α
      h : LE.le Top.top b
      this : Eq b Top.top
      ⊢ LE.le Top.top.prod b.prod
    -/
    rw [this, prod_top]
    /-
      🎉 no goals
    -/
                                                       /-
                                                         α : Type u_1
                                                         inst✝ : CancelCommMonoidWithZero α
                                                         a : Associates.FactorSet α
                                                         x✝ : LE.le a Top.top
                                                         ⊢ LE.le a.prod Top.top.prod
                                                       -/
  | a, ⊤, _ => show a.prod ≤ (⊤ : FactorSet α).prod by simp
                                                       /-
                                                         🎉 no goals
                                                       -/
  | WithTop.some _, WithTop.some _, h =>
    prod_le_prod <| Multiset.map_le_map <| WithTop.coe_le_coe.1 <| h


theorem FactorSet.prod_eq_zero_iff [Nontrivial α] (p : FactorSet α) : p.prod = 0 ↔ p = ⊤ := by
  /-
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : Nontrivial α
    p : Associates.FactorSet α
    ⊢ Iff (Eq p.prod 0) (Eq p Top.top)
  -/
  unfold FactorSet at p
  /-
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : Nontrivial α
    p : WithTop (Multiset (Subtype fun a => Irreducible a))
    ⊢ Iff (Eq (Associates.FactorSet.prod p) 0) (Eq p Top.top)
  -/
  induction p  -- TODO: `induction_eliminator` doesn't work with `abbrev`
    /-
      case top
      α : Type u_1
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : Nontrivial α
      ⊢ Iff (Eq (Associates.FactorSet.prod Top.top) 0) (Eq Top.top Top.top)
    -/
  · simp only [eq_self_iff_true, Associates.prod_top]
    /-
      🎉 no goals
    -/
  · rw [prod_coe, Multiset.prod_eq_zero_iff, Multiset.mem_map, eq_false WithTop.coe_ne_top,
      iff_false, not_exists]
    /-
      case coe
      α : Type u_1
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : Nontrivial α
      a✝ : Multiset (Subtype fun a => Irreducible a)
      ⊢ ∀ (x : Subtype fun a => Irreducible a), Not (And (Membership.mem a✝ x) (Eq ( …
    -/
    exact fun a => not_and_of_not_right _ a.prop.ne_zero
    /-
      🎉 no goals
    -/


/-- `bcount p s` is the multiplicity of `p` in the FactorSet `s` (with bundled `p`)-/
def bcount (p : { a : Associates α // Irreducible a }) :
    FactorSet α → ℕ
  | ⊤ => 0
  | WithTop.some s => s.count p


/-- `count p s` is the multiplicity of the irreducible `p` in the FactorSet `s`.

If `p` is not irreducible, `count p s` is defined to be `0`. -/
def count (p : Associates α) : FactorSet α → ℕ :=
  if hp : Irreducible p then bcount ⟨p, hp⟩ else 0


@[simp]
theorem count_some (hp : Irreducible p) (s : Multiset _) :
    count p (WithTop.some s) = s.count ⟨p, hp⟩ := by
  /-
    α : Type u_1
    inst✝² : CancelCommMonoidWithZero α
    inst✝¹ : DecidableEq (Associates α)
    inst✝ : (p : Associates α) → Decidable (Irreducible p)
    p : Associates α
    hp : Irreducible p
    s : Multiset (Subtype fun a => Irreducible a)
    ⊢ Eq (p.count ↑s) (Multiset.count ⟨p, hp⟩ s)
  -/
  simp only [count, dif_pos hp, bcount]
  /-
    🎉 no goals
  -/


@[simp]
theorem count_zero (hp : Irreducible p) : count p (0 : FactorSet α) = 0 := by
  /-
    α : Type u_1
    inst✝² : CancelCommMonoidWithZero α
    inst✝¹ : DecidableEq (Associates α)
    inst✝ : (p : Associates α) → Decidable (Irreducible p)
    p : Associates α
    hp : Irreducible p
    ⊢ Eq (p.count 0) 0
  -/
  simp only [count, dif_pos hp, bcount, Multiset.count_zero]
  /-
    🎉 no goals
  -/


theorem count_reducible (hp : ¬Irreducible p) : count p = 0 := dif_neg hp


/-- membership in a FactorSet (bundled version) -/
def BfactorSetMem : { a : Associates α // Irreducible a } → FactorSet α → Prop
  | _, ⊤ => True
  | p, some l => p ∈ l


/-- `FactorSetMem p s` is the predicate that the irreducible `p` is a member of
`s : FactorSet α`.

If `p` is not irreducible, `p` is not a member of any `FactorSet`. -/
def FactorSetMem (s : FactorSet α) (p : Associates α) : Prop :=
  letI : Decidable (Irreducible p) := Classical.dec _
  if hp : Irreducible p then BfactorSetMem ⟨p, hp⟩ s else False


instance : Membership (Associates α) (FactorSet α) :=
  ⟨FactorSetMem⟩


@[simp]
theorem factorSetMem_eq_mem (p : Associates α) (s : FactorSet α) : FactorSetMem s p = (p ∈ s) :=
  rfl


theorem mem_factorSet_top {p : Associates α} {hp : Irreducible p} : p ∈ (⊤ : FactorSet α) := by
  /-
    α : Type u_1
    inst✝ : CancelCommMonoidWithZero α
    p : Associates α
    hp : Irreducible p
    ⊢ Membership.mem Top.top p
  -/
  dsimp only [Membership.mem]; dsimp only [FactorSetMem]; split_ifs; exact trivial
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


theorem mem_factorSet_some {p : Associates α} {hp : Irreducible p}
    {l : Multiset { a : Associates α // Irreducible a }} :
    p ∈ (l : FactorSet α) ↔ Subtype.mk p hp ∈ l := by
  /-
    α : Type u_1
    inst✝ : CancelCommMonoidWithZero α
    p : Associates α
    hp : Irreducible p
    l : Multiset (Subtype fun a => Irreducible a)
    ⊢ Iff (Membership.mem (↑l) p) (Membership.mem l ⟨p, hp⟩)
  -/
  dsimp only [Membership.mem]; dsimp only [FactorSetMem]; split_ifs; rfl
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


theorem reducible_not_mem_factorSet {p : Associates α} (hp : ¬Irreducible p) (s : FactorSet α) :
    ¬p ∈ s := fun h ↦ by
  /-
    α : Type u_1
    inst✝ : CancelCommMonoidWithZero α
    p : Associates α
    hp : Not (Irreducible p)
    s : Associates.FactorSet α
    h : Membership.mem s p
    ⊢ False
  -/
  rwa [← factorSetMem_eq_mem, FactorSetMem, dif_neg hp] at h
  /-
    🎉 no goals
  -/


theorem irreducible_of_mem_factorSet {p : Associates α} {s : FactorSet α} (h : p ∈ s) :
    Irreducible p :=
  by_contra fun hp ↦ reducible_not_mem_factorSet hp s h


theorem FactorSet.unique [Nontrivial α] {p q : FactorSet α} (h : p.prod = q.prod) : p = q := by
  -- TODO: `induction_eliminator` doesn't work with `abbrev`
  /-
    α : Type u_1
    inst✝² : CancelCommMonoidWithZero α
    inst✝¹ : UniqueFactorizationMonoid α
    inst✝ : Nontrivial α
    p q : Associates.FactorSet α
    h : Eq p.prod q.prod
    ⊢ Eq p q
  -/
  unfold FactorSet at p q
  /-
    α : Type u_1
    inst✝² : CancelCommMonoidWithZero α
    inst✝¹ : UniqueFactorizationMonoid α
    inst✝ : Nontrivial α
    p q : WithTop (Multiset (Subtype fun a => Irreducible a))
    h : Eq (Associates.FactorSet.prod p) (Associates.FactorSet.prod q)
    ⊢ Eq p q
  -/
  induction p <;> induction q
    /-
      case top.top
      α : Type u_1
      inst✝² : CancelCommMonoidWithZero α
      inst✝¹ : UniqueFactorizationMonoid α
      inst✝ : Nontrivial α
      h : Eq (Associates.FactorSet.prod Top.top) (Associates.FactorSet.prod Top.top)
      ⊢ Eq Top.top Top.top
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case top.coe
      α : Type u_1
      inst✝² : CancelCommMonoidWithZero α
      inst✝¹ : UniqueFactorizationMonoid α
      inst✝ : Nontrivial α
      a✝ : Multiset (Subtype fun a => Irreducible a)
      h : Eq (Associates.FactorSet.prod Top.top) (Associates.FactorSet.prod ↑a✝)
      ⊢ Eq Top.top ↑a✝
    -/
  · rw [eq_comm, ← FactorSet.prod_eq_zero_iff, ← h, Associates.prod_top]
    /-
      🎉 no goals
    -/
    /-
      case coe.top
      α : Type u_1
      inst✝² : CancelCommMonoidWithZero α
      inst✝¹ : UniqueFactorizationMonoid α
      inst✝ : Nontrivial α
      a✝ : Multiset (Subtype fun a => Irreducible a)
      h : Eq (Associates.FactorSet.prod ↑a✝) (Associates.FactorSet.prod Top.top)
      ⊢ Eq (↑a✝) Top.top
    -/
  · rw [← FactorSet.prod_eq_zero_iff, h, Associates.prod_top]
    /-
      🎉 no goals
    -/
    /-
      case coe.coe
      α : Type u_1
      inst✝² : CancelCommMonoidWithZero α
      inst✝¹ : UniqueFactorizationMonoid α
      inst✝ : Nontrivial α
      a✝¹ a✝ : Multiset (Subtype fun a => Irreducible a)
      h : Eq (Associates.FactorSet.prod ↑a✝¹) (Associates.FactorSet.prod ↑a✝)
      ⊢ Eq ↑a✝¹ ↑a✝
    -/
  · congr 1
    /-
      case coe.coe.e_a
      α : Type u_1
      inst✝² : CancelCommMonoidWithZero α
      inst✝¹ : UniqueFactorizationMonoid α
      inst✝ : Nontrivial α
      a✝¹ a✝ : Multiset (Subtype fun a => Irreducible a)
      h : Eq (Associates.FactorSet.prod ↑a✝¹) (Associates.FactorSet.prod ↑a✝)
      ⊢ Eq a✝¹ a✝
    -/
    rw [← Multiset.map_eq_map Subtype.coe_injective]
    /-
      case coe.coe.e_a
      α : Type u_1
      inst✝² : CancelCommMonoidWithZero α
      inst✝¹ : UniqueFactorizationMonoid α
      inst✝ : Nontrivial α
      a✝¹ a✝ : Multiset (Subtype fun a => Irreducible a)
      h : Eq (Associates.FactorSet.prod ↑a✝¹) (Associates.FactorSet.prod ↑a✝)
      ⊢ Eq (Multiset.map (fun a => ↑a) a✝¹) (Multiset.map (fun a => ↑a) a✝)
    -/
    apply unique' _ _ h <;>
        /-
          α : Type u_1
          inst✝² : CancelCommMonoidWithZero α
          inst✝¹ : UniqueFactorizationMonoid α
          inst✝ : Nontrivial α
          a✝¹ a✝ : Multiset (Subtype fun a => Irreducible a)
          h : Eq (Associates.FactorSet.prod ↑a✝¹) (Associates.FactorSet.prod ↑a✝)
          ⊢ ∀ (a : Associates α), Membership.mem (Multiset.map Subtype.val a✝¹) a → Irre …
        -/
        /-
          α : Type u_1
          inst✝² : CancelCommMonoidWithZero α
          inst✝¹ : UniqueFactorizationMonoid α
          inst✝ : Nontrivial α
          a✝¹ a✝ : Multiset (Subtype fun a => Irreducible a)
          h : Eq (Associates.FactorSet.prod ↑a✝¹) (Associates.FactorSet.prod ↑a✝)
          a : Associates α
          ha : Membership.mem (Multiset.map Subtype.val a✝¹) a
          ⊢ Irreducible a
        -/
        /-
          case intro.mk.intro
          α : Type u_1
          inst✝² : CancelCommMonoidWithZero α
          inst✝¹ : UniqueFactorizationMonoid α
          inst✝ : Nontrivial α
          a✝¹ a✝ : Multiset (Subtype fun a => Irreducible a)
          h : Eq (Associates.FactorSet.prod ↑a✝¹) (Associates.FactorSet.prod ↑a✝)
          a' : Associates α
          irred : Irreducible a'
          ha : Membership.mem (Multiset.map Subtype.val a✝¹) ↑⟨a', irred⟩
          ⊢ Irreducible ↑⟨a', irred⟩
        -/
        /-
          🎉 no goals
        -/
        obtain ⟨⟨a', irred⟩, -, rfl⟩ := Multiset.mem_map.mp ha
        /-
          case intro.mk.intro
          α : Type u_1
          inst✝² : CancelCommMonoidWithZero α
          inst✝¹ : UniqueFactorizationMonoid α
          inst✝ : Nontrivial α
          a✝¹ a✝ : Multiset (Subtype fun a => Irreducible a)
          h : Eq (Associates.FactorSet.prod ↑a✝¹) (Associates.FactorSet.prod ↑a✝)
          a' : Associates α
          irred : Irreducible a'
          ha : Membership.mem (Multiset.map Subtype.val a✝) ↑⟨a', irred⟩
          ⊢ Irreducible ↑⟨a', irred⟩
        -/
        rwa [Subtype.coe_mk]
        /-
          🎉 no goals
        -/


/-- This returns the multiset of irreducible factors as a `FactorSet`,
  a multiset of irreducible associates `WithTop`. -/
noncomputable def factors' (a : α) : Multiset { a : Associates α // Irreducible a } :=
  (factors a).pmap (fun a ha => ⟨Associates.mk a, irreducible_mk.2 ha⟩) irreducible_of_factor


@[simp]
theorem map_subtype_coe_factors' {a : α} :
    (factors' a).map (↑) = (factors a).map Associates.mk := by
  /-
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : UniqueFactorizationMonoid α
    a : α
    ⊢ Eq (Multiset.map Subtype.val (Associates.factors' a)) (Multiset.map Associat …
  -/
  simp [factors', Multiset.map_pmap, Multiset.pmap_eq_map]
  /-
    🎉 no goals
  -/


theorem factors'_cong {a b : α} (h : a ~ᵤ b) : factors' a = factors' b := by
  /-
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : UniqueFactorizationMonoid α
    a b : α
    h : Associated a b
    ⊢ Eq (Associates.factors' a) (Associates.factors' b)
  -/
  obtain rfl | hb := eq_or_ne b 0
    /-
      case inl
      α : Type u_1
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : UniqueFactorizationMonoid α
      a : α
      h : Associated a 0
      ⊢ Eq (Associates.factors' a) (Associates.factors' 0)
    -/
  · rw [associated_zero_iff_eq_zero] at h
    /-
      case inl
      α : Type u_1
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : UniqueFactorizationMonoid α
      a : α
      h : Eq a 0
      ⊢ Eq (Associates.factors' a) (Associates.factors' 0)
    -/
    rw [h]
    /-
      🎉 no goals
    -/
  have ha : a ≠ 0 := by
    contrapose! hb with ha
    rw [← associated_zero_iff_eq_zero, ← ha]
    exact h.symm
  rw [← Multiset.map_eq_map Subtype.coe_injective, map_subtype_coe_factors',
    map_subtype_coe_factors', ← rel_associated_iff_map_eq_map]
  exact
    factors_unique irreducible_of_factor irreducible_of_factor
      ((factors_prod ha).trans <| h.trans <| (factors_prod hb).symm)


/-- This returns the multiset of irreducible factors of an associate as a `FactorSet`,
  a multiset of irreducible associates `WithTop`. -/
noncomputable def factors (a : Associates α) : FactorSet α := by
  /-
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : UniqueFactorizationMonoid α
    a : Associates α
    ⊢ Associates.FactorSet α
  -/
  classical refine if h : a = 0 then ⊤ else Quotient.hrecOn a (fun x _ => factors' x) ?_ h
  /-
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : UniqueFactorizationMonoid α
    a : Associates α
    h : Not (Eq a 0)
    ⊢ ∀ (a b : α), HasEquiv.Equiv a b → HEq (fun x => ↑(Associates.factors' a)) fu …
  -/
  intro a b hab
  /-
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : UniqueFactorizationMonoid α
    a✝ : Associates α
    h : Not (Eq a✝ 0)
    a b : α
    hab : HasEquiv.Equiv a b
    ⊢ HEq (fun x => ↑(Associates.factors' a)) fun x => ↑(Associates.factors' b)
  -/
  apply Function.hfunext
    /-
      case hα
      α : Type u_1
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : UniqueFactorizationMonoid α
      a✝ : Associates α
      h : Not (Eq a✝ 0)
      a b : α
      hab : HasEquiv.Equiv a b
      ⊢ Eq (Not (Eq (Quotient.mk (Associated.setoid α) a) 0)) (Not (Eq (Quotient.mk  …
    -/
  · have : a ~ᵤ 0 ↔ b ~ᵤ 0 := Iff.intro (fun ha0 => hab.symm.trans ha0) fun hb0 => hab.trans hb0
    /-
      case hα
      α : Type u_1
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : UniqueFactorizationMonoid α
      a✝ : Associates α
      h : Not (Eq a✝ 0)
      a b : α
      hab : HasEquiv.Equiv a b
      this : Iff (Associated a 0) (Associated b 0)
      ⊢ Eq (Not (Eq (Quotient.mk (Associated.setoid α) a) 0)) (Not (Eq (Quotient.mk  …
    -/
    simp only [associated_zero_iff_eq_zero] at this
    /-
      case hα
      α : Type u_1
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : UniqueFactorizationMonoid α
      a✝ : Associates α
      h : Not (Eq a✝ 0)
      a b : α
      hab : HasEquiv.Equiv a b
      this : Iff (Eq a 0) (Eq b 0)
      ⊢ Eq (Not (Eq (Quotient.mk (Associated.setoid α) a) 0)) (Not (Eq (Quotient.mk  …
    -/
    simp only [quotient_mk_eq_mk, this, mk_eq_zero]
    /-
      🎉 no goals
    -/
  /-
    case h
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : UniqueFactorizationMonoid α
    a✝ : Associates α
    h : Not (Eq a✝ 0)
    a b : α
    hab : HasEquiv.Equiv a b
    ⊢ ∀ (a_1 : Not (Eq (Quotient.mk (Associated.setoid α) a) 0)) (a' : Not (Eq (Qu …
  -/
  exact fun ha hb _ => heq_of_eq <| congr_arg some <| factors'_cong hab
  /-
    🎉 no goals
  -/


@[simp]
theorem factors_zero : (0 : Associates α).factors = ⊤ :=
  dif_pos rfl


@[deprecated (since := "2024-03-16")] alias factors_0 := factors_zero


@[simp]
theorem factors_mk (a : α) (h : a ≠ 0) : (Associates.mk a).factors = factors' a := by
  classical
    apply dif_neg
    apply mt mk_eq_zero.1 h


@[simp]
theorem factors_prod (a : Associates α) : a.factors.prod = a := by
  /-
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : UniqueFactorizationMonoid α
    a : Associates α
    ⊢ Eq a.factors.prod a
  -/
  rcases Associates.mk_surjective a with ⟨a, rfl⟩
  /-
    case intro
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : UniqueFactorizationMonoid α
    a : α
    ⊢ Eq (Associates.mk a).factors.prod (Associates.mk a)
  -/
  rcases eq_or_ne a 0 with rfl | ha
    /-
      case intro.inl
      α : Type u_1
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : UniqueFactorizationMonoid α
      ⊢ Eq (Associates.mk 0).factors.prod (Associates.mk 0)
    -/
  · simp
    /-
      🎉 no goals
    -/
  · simp [ha, prod_mk, mk_eq_mk_iff_associated, UniqueFactorizationMonoid.factors_prod,
      -Quotient.eq]


@[simp]
theorem prod_factors [Nontrivial α] (s : FactorSet α) : s.prod.factors = s :=
  FactorSet.unique <| factors_prod _


@[nontriviality]
theorem factors_subsingleton [Subsingleton α] {a : Associates α} : a.factors = ⊤ := by
  /-
    α : Type u_1
    inst✝² : CancelCommMonoidWithZero α
    inst✝¹ : UniqueFactorizationMonoid α
    inst✝ : Subsingleton α
    a : Associates α
    ⊢ Eq a.factors Top.top
  -/
  have : Subsingleton (Associates α) := inferInstance
  /-
    α : Type u_1
    inst✝² : CancelCommMonoidWithZero α
    inst✝¹ : UniqueFactorizationMonoid α
    inst✝ : Subsingleton α
    a : Associates α
    this : Subsingleton (Associates α)
    ⊢ Eq a.factors Top.top
  -/
  convert factors_zero
  /-
    🎉 no goals
  -/


theorem factors_eq_top_iff_zero {a : Associates α} : a.factors = ⊤ ↔ a = 0 := by
  /-
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : UniqueFactorizationMonoid α
    a : Associates α
    ⊢ Iff (Eq a.factors Top.top) (Eq a 0)
  -/
  nontriviality α
  /-
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : UniqueFactorizationMonoid α
    a : Associates α
    a✝ : Nontrivial α
    ⊢ Iff (Eq a.factors Top.top) (Eq a 0)
  -/
  exact ⟨fun h ↦ by rwa [← factors_prod a, FactorSet.prod_eq_zero_iff], fun h ↦ h ▸ factors_zero⟩
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-04-16")] alias factors_eq_none_iff_zero := factors_eq_top_iff_zero


theorem factors_eq_some_iff_ne_zero {a : Associates α} :
    (∃ s : Multiset { p : Associates α // Irreducible p }, a.factors = s) ↔ a ≠ 0 := by
  /-
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : UniqueFactorizationMonoid α
    a : Associates α
    ⊢ Iff (Exists fun s => Eq a.factors ↑s) (Ne a 0)
  -/
  simp_rw [@eq_comm _ a.factors, ← WithTop.ne_top_iff_exists]
  /-
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : UniqueFactorizationMonoid α
    a : Associates α
    ⊢ Iff (Ne a.factors Top.top) (Ne a 0)
  -/
  exact factors_eq_top_iff_zero.not
  /-
    🎉 no goals
  -/


theorem eq_of_factors_eq_factors {a b : Associates α} (h : a.factors = b.factors) : a = b := by
  /-
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : UniqueFactorizationMonoid α
    a b : Associates α
    h : Eq a.factors b.factors
    ⊢ Eq a b
  -/
  have : a.factors.prod = b.factors.prod := by rw [h]
  /-
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : UniqueFactorizationMonoid α
    a b : Associates α
    h : Eq a.factors b.factors
    this : Eq a.factors.prod b.factors.prod
    ⊢ Eq a b
  -/
  rwa [factors_prod, factors_prod] at this
  /-
    🎉 no goals
  -/


theorem eq_of_prod_eq_prod [Nontrivial α] {a b : FactorSet α} (h : a.prod = b.prod) : a = b := by
  /-
    α : Type u_1
    inst✝² : CancelCommMonoidWithZero α
    inst✝¹ : UniqueFactorizationMonoid α
    inst✝ : Nontrivial α
    a b : Associates.FactorSet α
    h : Eq a.prod b.prod
    ⊢ Eq a b
  -/
  have : a.prod.factors = b.prod.factors := by rw [h]
  /-
    α : Type u_1
    inst✝² : CancelCommMonoidWithZero α
    inst✝¹ : UniqueFactorizationMonoid α
    inst✝ : Nontrivial α
    a b : Associates.FactorSet α
    h : Eq a.prod b.prod
    this : Eq a.prod.factors b.prod.factors
    ⊢ Eq a b
  -/
  rwa [prod_factors, prod_factors] at this
  /-
    🎉 no goals
  -/


@[simp]
theorem factors_mul (a b : Associates α) : (a * b).factors = a.factors + b.factors := by
  /-
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : UniqueFactorizationMonoid α
    a b : Associates α
    ⊢ Eq (HMul.hMul a b).factors (HAdd.hAdd a.factors b.factors)
  -/
  nontriviality α
  /-
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : UniqueFactorizationMonoid α
    a b : Associates α
    a✝ : Nontrivial α
    ⊢ Eq (HMul.hMul a b).factors (HAdd.hAdd a.factors b.factors)
  -/
  refine eq_of_prod_eq_prod <| eq_of_factors_eq_factors ?_
  /-
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : UniqueFactorizationMonoid α
    a b : Associates α
    a✝ : Nontrivial α
    ⊢ Eq (HMul.hMul a b).factors.prod.factors (HAdd.hAdd a.factors b.factors).prod …
  -/
  rw [prod_add, factors_prod, factors_prod, factors_prod]
  /-
    🎉 no goals
  -/


@[gcongr]
theorem factors_mono : ∀ {a b : Associates α}, a ≤ b → a.factors ≤ b.factors
                        /-
                          α : Type u_1
                          inst✝¹ : CancelCommMonoidWithZero α
                          inst✝ : UniqueFactorizationMonoid α
                          s t : Associates α
                          d : Associates α
                          eq : Eq t (HMul.hMul s d)
                          ⊢ LE.le s.factors t.factors
                        -/
  | s, t, ⟨d, eq⟩ => by rw [eq, factors_mul]; exact le_add_of_nonneg_right bot_le
                                              /-
                                                🎉 no goals
                                              -/


@[simp]
theorem factors_le {a b : Associates α} : a.factors ≤ b.factors ↔ a ≤ b := by
  /-
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : UniqueFactorizationMonoid α
    a b : Associates α
    ⊢ Iff (LE.le a.factors b.factors) (LE.le a b)
  -/
  refine ⟨fun h ↦ ?_, factors_mono⟩
  /-
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : UniqueFactorizationMonoid α
    a b : Associates α
    h : LE.le a.factors b.factors
    ⊢ LE.le a b
  -/
  have : a.factors.prod ≤ b.factors.prod := prod_mono h
  /-
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : UniqueFactorizationMonoid α
    a b : Associates α
    h : LE.le a.factors b.factors
    this : LE.le a.factors.prod b.factors.prod
    ⊢ LE.le a b
  -/
  rwa [factors_prod, factors_prod] at this
  /-
    🎉 no goals
  -/


theorem eq_factors_of_eq_counts {a b : Associates α} (ha : a ≠ 0) (hb : b ≠ 0)
    (h : ∀ p : Associates α, Irreducible p → p.count a.factors = p.count b.factors) :
    a.factors = b.factors := by
  /-
    α : Type u_1
    inst✝³ : CancelCommMonoidWithZero α
    inst✝² : UniqueFactorizationMonoid α
    inst✝¹ : DecidableEq (Associates α)
    inst✝ : (p : Associates α) → Decidable (Irreducible p)
    a b : Associates α
    ha : Ne a 0
    hb : Ne b 0
    h : ∀ (p : Associates α), Irreducible p → Eq (p.count a.factors) (p.count b.fa …
    ⊢ Eq a.factors b.factors
  -/
  obtain ⟨sa, h_sa⟩ := factors_eq_some_iff_ne_zero.mpr ha
  /-
    case intro
    α : Type u_1
    inst✝³ : CancelCommMonoidWithZero α
    inst✝² : UniqueFactorizationMonoid α
    inst✝¹ : DecidableEq (Associates α)
    inst✝ : (p : Associates α) → Decidable (Irreducible p)
    a b : Associates α
    ha : Ne a 0
    hb : Ne b 0
    h : ∀ (p : Associates α), Irreducible p → Eq (p.count a.factors) (p.count b.fa …
    sa : Multiset (Subtype fun p => Irreducible p)
    h_sa : Eq a.factors ↑sa
    ⊢ Eq a.factors b.factors
  -/
  obtain ⟨sb, h_sb⟩ := factors_eq_some_iff_ne_zero.mpr hb
  /-
    case intro.intro
    α : Type u_1
    inst✝³ : CancelCommMonoidWithZero α
    inst✝² : UniqueFactorizationMonoid α
    inst✝¹ : DecidableEq (Associates α)
    inst✝ : (p : Associates α) → Decidable (Irreducible p)
    a b : Associates α
    ha : Ne a 0
    hb : Ne b 0
    h : ∀ (p : Associates α), Irreducible p → Eq (p.count a.factors) (p.count b.fa …
    sa : Multiset (Subtype fun p => Irreducible p)
    h_sa : Eq a.factors ↑sa
    sb : Multiset (Subtype fun p => Irreducible p)
    h_sb : Eq b.factors ↑sb
    ⊢ Eq a.factors b.factors
  -/
  rw [h_sa, h_sb] at h ⊢
  /-
    case intro.intro
    α : Type u_1
    inst✝³ : CancelCommMonoidWithZero α
    inst✝² : UniqueFactorizationMonoid α
    inst✝¹ : DecidableEq (Associates α)
    inst✝ : (p : Associates α) → Decidable (Irreducible p)
    a b : Associates α
    ha : Ne a 0
    hb : Ne b 0
    sa : Multiset (Subtype fun p => Irreducible p)
    h_sa : Eq a.factors ↑sa
    sb : Multiset (Subtype fun p => Irreducible p)
    h : ∀ (p : Associates α), Irreducible p → Eq (p.count ↑sa) (p.count ↑sb)
    h_sb : Eq b.factors ↑sb
    ⊢ Eq ↑sa ↑sb
  -/
  rw [WithTop.coe_eq_coe]
  have h_count : ∀ (p : Associates α) (hp : Irreducible p),
      sa.count ⟨p, hp⟩ = sb.count ⟨p, hp⟩ := by
    intro p hp
    rw [← count_some, ← count_some, h p hp]
  /-
    case intro.intro
    α : Type u_1
    inst✝³ : CancelCommMonoidWithZero α
    inst✝² : UniqueFactorizationMonoid α
    inst✝¹ : DecidableEq (Associates α)
    inst✝ : (p : Associates α) → Decidable (Irreducible p)
    a b : Associates α
    ha : Ne a 0
    hb : Ne b 0
    sa : Multiset (Subtype fun p => Irreducible p)
    h_sa : Eq a.factors ↑sa
    sb : Multiset (Subtype fun p => Irreducible p)
    h : ∀ (p : Associates α), Irreducible p → Eq (p.count ↑sa) (p.count ↑sb)
    h_sb : Eq b.factors ↑sb
    h_count : ∀ (p : Associates α) (hp : Irreducible p), Eq (Multiset.count ⟨p, hp …
    ⊢ Eq sa sb
  -/
  apply Multiset.toFinsupp.injective
  /-
    case intro.intro.a
    α : Type u_1
    inst✝³ : CancelCommMonoidWithZero α
    inst✝² : UniqueFactorizationMonoid α
    inst✝¹ : DecidableEq (Associates α)
    inst✝ : (p : Associates α) → Decidable (Irreducible p)
    a b : Associates α
    ha : Ne a 0
    hb : Ne b 0
    sa : Multiset (Subtype fun p => Irreducible p)
    h_sa : Eq a.factors ↑sa
    sb : Multiset (Subtype fun p => Irreducible p)
    h : ∀ (p : Associates α), Irreducible p → Eq (p.count ↑sa) (p.count ↑sb)
    h_sb : Eq b.factors ↑sb
    h_count : ∀ (p : Associates α) (hp : Irreducible p), Eq (Multiset.count ⟨p, hp …
    ⊢ Eq (Multiset.toFinsupp sa) (Multiset.toFinsupp sb)
  -/
  ext ⟨p, hp⟩
  /-
    case intro.intro.a.h.mk
    α : Type u_1
    inst✝³ : CancelCommMonoidWithZero α
    inst✝² : UniqueFactorizationMonoid α
    inst✝¹ : DecidableEq (Associates α)
    inst✝ : (p : Associates α) → Decidable (Irreducible p)
    a b : Associates α
    ha : Ne a 0
    hb : Ne b 0
    sa : Multiset (Subtype fun p => Irreducible p)
    h_sa : Eq a.factors ↑sa
    sb : Multiset (Subtype fun p => Irreducible p)
    h : ∀ (p : Associates α), Irreducible p → Eq (p.count ↑sa) (p.count ↑sb)
    h_sb : Eq b.factors ↑sb
    h_count : ∀ (p : Associates α) (hp : Irreducible p), Eq (Multiset.count ⟨p, hp …
    p : Associates α
    hp : Irreducible p
    ⊢ Eq ((Multiset.toFinsupp sa) ⟨p, hp⟩) ((Multiset.toFinsupp sb) ⟨p, hp⟩)
  -/
  rw [Multiset.toFinsupp_apply, Multiset.toFinsupp_apply, h_count p hp]
  /-
    🎉 no goals
  -/


theorem eq_of_eq_counts {a b : Associates α} (ha : a ≠ 0) (hb : b ≠ 0)
    (h : ∀ p : Associates α, Irreducible p → p.count a.factors = p.count b.factors) : a = b :=
  eq_of_factors_eq_factors (eq_factors_of_eq_counts ha hb h)


theorem count_le_count_of_factors_le {a b p : Associates α} (hb : b ≠ 0) (hp : Irreducible p)
    (h : a.factors ≤ b.factors) : p.count a.factors ≤ p.count b.factors := by
  /-
    α : Type u_1
    inst✝³ : CancelCommMonoidWithZero α
    inst✝² : UniqueFactorizationMonoid α
    inst✝¹ : DecidableEq (Associates α)
    inst✝ : (p : Associates α) → Decidable (Irreducible p)
    a b p : Associates α
    hb : Ne b 0
    hp : Irreducible p
    h : LE.le a.factors b.factors
    ⊢ LE.le (p.count a.factors) (p.count b.factors)
  -/
  by_cases ha : a = 0
    /-
      case pos
      α : Type u_1
      inst✝³ : CancelCommMonoidWithZero α
      inst✝² : UniqueFactorizationMonoid α
      inst✝¹ : DecidableEq (Associates α)
      inst✝ : (p : Associates α) → Decidable (Irreducible p)
      a b p : Associates α
      hb : Ne b 0
      hp : Irreducible p
      h : LE.le a.factors b.factors
      ha : Eq a 0
      ⊢ LE.le (p.count a.factors) (p.count b.factors)
    -/
  · simp_all
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    inst✝³ : CancelCommMonoidWithZero α
    inst✝² : UniqueFactorizationMonoid α
    inst✝¹ : DecidableEq (Associates α)
    inst✝ : (p : Associates α) → Decidable (Irreducible p)
    a b p : Associates α
    hb : Ne b 0
    hp : Irreducible p
    h : LE.le a.factors b.factors
    ha : Not (Eq a 0)
    ⊢ LE.le (p.count a.factors) (p.count b.factors)
  -/
  obtain ⟨sa, h_sa⟩ := factors_eq_some_iff_ne_zero.mpr ha
  /-
    case neg.intro
    α : Type u_1
    inst✝³ : CancelCommMonoidWithZero α
    inst✝² : UniqueFactorizationMonoid α
    inst✝¹ : DecidableEq (Associates α)
    inst✝ : (p : Associates α) → Decidable (Irreducible p)
    a b p : Associates α
    hb : Ne b 0
    hp : Irreducible p
    h : LE.le a.factors b.factors
    ha : Not (Eq a 0)
    sa : Multiset (Subtype fun p => Irreducible p)
    h_sa : Eq a.factors ↑sa
    ⊢ LE.le (p.count a.factors) (p.count b.factors)
  -/
  obtain ⟨sb, h_sb⟩ := factors_eq_some_iff_ne_zero.mpr hb
  /-
    case neg.intro.intro
    α : Type u_1
    inst✝³ : CancelCommMonoidWithZero α
    inst✝² : UniqueFactorizationMonoid α
    inst✝¹ : DecidableEq (Associates α)
    inst✝ : (p : Associates α) → Decidable (Irreducible p)
    a b p : Associates α
    hb : Ne b 0
    hp : Irreducible p
    h : LE.le a.factors b.factors
    ha : Not (Eq a 0)
    sa : Multiset (Subtype fun p => Irreducible p)
    h_sa : Eq a.factors ↑sa
    sb : Multiset (Subtype fun p => Irreducible p)
    h_sb : Eq b.factors ↑sb
    ⊢ LE.le (p.count a.factors) (p.count b.factors)
  -/
  rw [h_sa, h_sb] at h ⊢
  /-
    case neg.intro.intro
    α : Type u_1
    inst✝³ : CancelCommMonoidWithZero α
    inst✝² : UniqueFactorizationMonoid α
    inst✝¹ : DecidableEq (Associates α)
    inst✝ : (p : Associates α) → Decidable (Irreducible p)
    a b p : Associates α
    hb : Ne b 0
    hp : Irreducible p
    ha : Not (Eq a 0)
    sa : Multiset (Subtype fun p => Irreducible p)
    h_sa : Eq a.factors ↑sa
    sb : Multiset (Subtype fun p => Irreducible p)
    h : LE.le ↑sa ↑sb
    h_sb : Eq b.factors ↑sb
    ⊢ LE.le (p.count ↑sa) (p.count ↑sb)
  -/
  rw [count_some hp, count_some hp]; rw [WithTop.coe_le_coe] at h
  /-
    case neg.intro.intro
    α : Type u_1
    inst✝³ : CancelCommMonoidWithZero α
    inst✝² : UniqueFactorizationMonoid α
    inst✝¹ : DecidableEq (Associates α)
    inst✝ : (p : Associates α) → Decidable (Irreducible p)
    a b p : Associates α
    hb : Ne b 0
    hp : Irreducible p
    ha : Not (Eq a 0)
    sa : Multiset (Subtype fun p => Irreducible p)
    h_sa : Eq a.factors ↑sa
    sb : Multiset (Subtype fun p => Irreducible p)
    h : LE.le sa sb
    h_sb : Eq b.factors ↑sb
    ⊢ LE.le (Multiset.count ⟨p, hp⟩ sa) (Multiset.count ⟨p, hp⟩ sb)
  -/
  exact Multiset.count_le_of_le _ h
  /-
    🎉 no goals
  -/


theorem count_le_count_of_le {a b p : Associates α} (hb : b ≠ 0) (hp : Irreducible p) (h : a ≤ b) :
    p.count a.factors ≤ p.count b.factors :=
  count_le_count_of_factors_le hb hp <| factors_mono h


theorem prod_le [Nontrivial α] {a b : FactorSet α} : a.prod ≤ b.prod ↔ a ≤ b := by
  /-
    α : Type u_1
    inst✝² : CancelCommMonoidWithZero α
    inst✝¹ : UniqueFactorizationMonoid α
    inst✝ : Nontrivial α
    a b : Associates.FactorSet α
    ⊢ Iff (LE.le a.prod b.prod) (LE.le a b)
  -/
  refine ⟨fun h ↦ ?_, prod_mono⟩
  /-
    α : Type u_1
    inst✝² : CancelCommMonoidWithZero α
    inst✝¹ : UniqueFactorizationMonoid α
    inst✝ : Nontrivial α
    a b : Associates.FactorSet α
    h : LE.le a.prod b.prod
    ⊢ LE.le a b
  -/
  have : a.prod.factors ≤ b.prod.factors := factors_mono h
  /-
    α : Type u_1
    inst✝² : CancelCommMonoidWithZero α
    inst✝¹ : UniqueFactorizationMonoid α
    inst✝ : Nontrivial α
    a b : Associates.FactorSet α
    h : LE.le a.prod b.prod
    this : LE.le a.prod.factors b.prod.factors
    ⊢ LE.le a b
  -/
  rwa [prod_factors, prod_factors] at this
  /-
    🎉 no goals
  -/


open Classical in
noncomputable instance : Max (Associates α) :=
  ⟨fun a b => (a.factors ⊔ b.factors).prod⟩


open Classical in
noncomputable instance : Min (Associates α) :=
  ⟨fun a b => (a.factors ⊓ b.factors).prod⟩


open Classical in
noncomputable instance : Lattice (Associates α) :=
  { Associates.instPartialOrder with
    sup := (· ⊔ ·)
    inf := (· ⊓ ·)
    sup_le := fun _ _ c hac hbc =>
      factors_prod c ▸ prod_mono (sup_le (factors_mono hac) (factors_mono hbc))
    le_sup_left := fun a _ => le_trans (le_of_eq (factors_prod a).symm) <| prod_mono <| le_sup_left
    le_sup_right := fun _ b =>
      le_trans (le_of_eq (factors_prod b).symm) <| prod_mono <| le_sup_right
    le_inf := fun a _ _ hac hbc =>
      factors_prod a ▸ prod_mono (le_inf (factors_mono hac) (factors_mono hbc))
    inf_le_left := fun a _ => le_trans (prod_mono inf_le_left) (le_of_eq (factors_prod a))
    inf_le_right := fun _ b => le_trans (prod_mono inf_le_right) (le_of_eq (factors_prod b)) }


open Classical in
theorem sup_mul_inf (a b : Associates α) : (a ⊔ b) * (a ⊓ b) = a * b :=
  show (a.factors ⊔ b.factors).prod * (a.factors ⊓ b.factors).prod = a * b by
    /-
      α : Type u_1
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : UniqueFactorizationMonoid α
      a b : Associates α
      ⊢ Eq (HMul.hMul (Max.max a.factors b.factors).prod (Min.min a.factors b.factor …
    -/
    nontriviality α
    /-
      α : Type u_1
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : UniqueFactorizationMonoid α
      a b : Associates α
      a✝ : Nontrivial α
      ⊢ Eq (HMul.hMul (Max.max a.factors b.factors).prod (Min.min a.factors b.factor …
    -/
    refine eq_of_factors_eq_factors ?_
    /-
      α : Type u_1
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : UniqueFactorizationMonoid α
      a b : Associates α
      a✝ : Nontrivial α
      ⊢ Eq (HMul.hMul (Max.max a.factors b.factors).prod (Min.min a.factors b.factor …
    -/
    rw [← prod_add, prod_factors, factors_mul, FactorSet.sup_add_inf_eq_add]
    /-
      🎉 no goals
    -/


theorem dvd_of_mem_factors {a p : Associates α} (hm : p ∈ factors a) :
    p ∣ a := by
  /-
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : UniqueFactorizationMonoid α
    a p : Associates α
    hm : Membership.mem a.factors p
    ⊢ Dvd.dvd p a
  -/
  rcases eq_or_ne a 0 with rfl | ha0
    /-
      case inl
      α : Type u_1
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : UniqueFactorizationMonoid α
      p : Associates α
      hm : Membership.mem (Associates.factors 0) p
      ⊢ Dvd.dvd p 0
    -/
  · exact dvd_zero p
    /-
      🎉 no goals
    -/
  /-
    case inr
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : UniqueFactorizationMonoid α
    a p : Associates α
    hm : Membership.mem a.factors p
    ha0 : Ne a 0
    ⊢ Dvd.dvd p a
  -/
  obtain ⟨a0, nza, ha'⟩ := exists_non_zero_rep ha0
  /-
    case inr.intro.intro
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : UniqueFactorizationMonoid α
    a p : Associates α
    hm : Membership.mem a.factors p
    ha0 : Ne a 0
    a0 : α
    nza : Ne a0 0
    ha' : Eq (Associates.mk a0) a
    ⊢ Dvd.dvd p a
  -/
  rw [← Associates.factors_prod a]
  /-
    case inr.intro.intro
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : UniqueFactorizationMonoid α
    a p : Associates α
    hm : Membership.mem a.factors p
    ha0 : Ne a 0
    a0 : α
    nza : Ne a0 0
    ha' : Eq (Associates.mk a0) a
    ⊢ Dvd.dvd p a.factors.prod
  -/
  rw [← ha', factors_mk a0 nza] at hm ⊢
  /-
    case inr.intro.intro
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : UniqueFactorizationMonoid α
    a p : Associates α
    ha0 : Ne a 0
    a0 : α
    hm : Membership.mem (↑(Associates.factors' a0)) p
    nza : Ne a0 0
    ha' : Eq (Associates.mk a0) a
    ⊢ Dvd.dvd p (Associates.FactorSet.prod ↑(Associates.factors' a0))
  -/
  rw [prod_coe]
  /-
    case inr.intro.intro
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : UniqueFactorizationMonoid α
    a p : Associates α
    ha0 : Ne a 0
    a0 : α
    hm : Membership.mem (↑(Associates.factors' a0)) p
    nza : Ne a0 0
    ha' : Eq (Associates.mk a0) a
    ⊢ Dvd.dvd p (Multiset.map Subtype.val (Associates.factors' a0)).prod
  -/
  apply Multiset.dvd_prod; apply Multiset.mem_map.mpr
  /-
    case inr.intro.intro.a
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : UniqueFactorizationMonoid α
    a p : Associates α
    ha0 : Ne a 0
    a0 : α
    hm : Membership.mem (↑(Associates.factors' a0)) p
    nza : Ne a0 0
    ha' : Eq (Associates.mk a0) a
    ⊢ Exists fun a => And (Membership.mem (Associates.factors' a0) a) (Eq (↑a) p)
  -/
  exact ⟨⟨p, irreducible_of_mem_factorSet hm⟩, mem_factorSet_some.mp hm, rfl⟩
  /-
    🎉 no goals
  -/


theorem dvd_of_mem_factors' {a : α} {p : Associates α} {hp : Irreducible p} {hz : a ≠ 0}
    (h_mem : Subtype.mk p hp ∈ factors' a) : p ∣ Associates.mk a := by
  /-
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : UniqueFactorizationMonoid α
    a : α
    p : Associates α
    hp : Irreducible p
    hz : Ne a 0
    h_mem : Membership.mem (Associates.factors' a) ⟨p, hp⟩
    ⊢ Dvd.dvd p (Associates.mk a)
  -/
  haveI := Classical.decEq (Associates α)
  /-
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : UniqueFactorizationMonoid α
    a : α
    p : Associates α
    hp : Irreducible p
    hz : Ne a 0
    h_mem : Membership.mem (Associates.factors' a) ⟨p, hp⟩
    this : DecidableEq (Associates α)
    ⊢ Dvd.dvd p (Associates.mk a)
  -/
  apply dvd_of_mem_factors
  /-
    case hm
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : UniqueFactorizationMonoid α
    a : α
    p : Associates α
    hp : Irreducible p
    hz : Ne a 0
    h_mem : Membership.mem (Associates.factors' a) ⟨p, hp⟩
    this : DecidableEq (Associates α)
    ⊢ Membership.mem (Associates.mk a).factors p
  -/
  rw [factors_mk _ hz]
  /-
    case hm
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : UniqueFactorizationMonoid α
    a : α
    p : Associates α
    hp : Irreducible p
    hz : Ne a 0
    h_mem : Membership.mem (Associates.factors' a) ⟨p, hp⟩
    this : DecidableEq (Associates α)
    ⊢ Membership.mem (↑(Associates.factors' a)) p
  -/
  apply mem_factorSet_some.2 h_mem
  /-
    🎉 no goals
  -/


theorem mem_factors'_of_dvd {a p : α} (ha0 : a ≠ 0) (hp : Irreducible p) (hd : p ∣ a) :
    Subtype.mk (Associates.mk p) (irreducible_mk.2 hp) ∈ factors' a := by
  /-
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : UniqueFactorizationMonoid α
    a p : α
    ha0 : Ne a 0
    hp : Irreducible p
    hd : Dvd.dvd p a
    ⊢ Membership.mem (Associates.factors' a) ⟨Associates.mk p, ⋯⟩
  -/
  obtain ⟨q, hq, hpq⟩ := exists_mem_factors_of_dvd ha0 hp hd
  /-
    case intro.intro
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : UniqueFactorizationMonoid α
    a p : α
    ha0 : Ne a 0
    hp : Irreducible p
    hd : Dvd.dvd p a
    q : α
    hq : Membership.mem (UniqueFactorizationMonoid.factors a) q
    hpq : Associated p q
    ⊢ Membership.mem (Associates.factors' a) ⟨Associates.mk p, ⋯⟩
  -/
  apply Multiset.mem_pmap.mpr; use q; use hq
  /-
    case h
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : UniqueFactorizationMonoid α
    a p : α
    ha0 : Ne a 0
    hp : Irreducible p
    hd : Dvd.dvd p a
    q : α
    hq : Membership.mem (UniqueFactorizationMonoid.factors a) q
    hpq : Associated p q
    ⊢ Eq ⟨Associates.mk q, ⋯⟩ ⟨Associates.mk p, ⋯⟩
  -/
  exact Subtype.eq (Eq.symm (mk_eq_mk_iff_associated.mpr hpq))
  /-
    🎉 no goals
  -/


theorem mem_factors'_iff_dvd {a p : α} (ha0 : a ≠ 0) (hp : Irreducible p) :
    Subtype.mk (Associates.mk p) (irreducible_mk.2 hp) ∈ factors' a ↔ p ∣ a := by
  /-
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : UniqueFactorizationMonoid α
    a p : α
    ha0 : Ne a 0
    hp : Irreducible p
    ⊢ Iff (Membership.mem (Associates.factors' a) ⟨Associates.mk p, ⋯⟩) (Dvd.dvd p …
  -/
  constructor
    /-
      case mp
      α : Type u_1
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : UniqueFactorizationMonoid α
      a p : α
      ha0 : Ne a 0
      hp : Irreducible p
      ⊢ Membership.mem (Associates.factors' a) ⟨Associates.mk p, ⋯⟩ → Dvd.dvd p a
    -/
  · rw [← mk_dvd_mk]
    /-
      case mp
      α : Type u_1
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : UniqueFactorizationMonoid α
      a p : α
      ha0 : Ne a 0
      hp : Irreducible p
      ⊢ Membership.mem (Associates.factors' a) ⟨Associates.mk p, ⋯⟩ → Dvd.dvd (Assoc …
    -/
    apply dvd_of_mem_factors'
    /-
      case mp.hz
      α : Type u_1
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : UniqueFactorizationMonoid α
      a p : α
      ha0 : Ne a 0
      hp : Irreducible p
      ⊢ Ne a 0
    -/
    apply ha0
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u_1
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : UniqueFactorizationMonoid α
      a p : α
      ha0 : Ne a 0
      hp : Irreducible p
      ⊢ Dvd.dvd p a → Membership.mem (Associates.factors' a) ⟨Associates.mk p, ⋯⟩
    -/
  · apply mem_factors'_of_dvd ha0 hp
    /-
      🎉 no goals
    -/


theorem mem_factors_of_dvd {a p : α} (ha0 : a ≠ 0) (hp : Irreducible p) (hd : p ∣ a) :
    Associates.mk p ∈ factors (Associates.mk a) := by
  /-
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : UniqueFactorizationMonoid α
    a p : α
    ha0 : Ne a 0
    hp : Irreducible p
    hd : Dvd.dvd p a
    ⊢ Membership.mem (Associates.mk a).factors (Associates.mk p)
  -/
  rw [factors_mk _ ha0]
  /-
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : UniqueFactorizationMonoid α
    a p : α
    ha0 : Ne a 0
    hp : Irreducible p
    hd : Dvd.dvd p a
    ⊢ Membership.mem (↑(Associates.factors' a)) (Associates.mk p)
  -/
  exact mem_factorSet_some.mpr (mem_factors'_of_dvd ha0 hp hd)
  /-
    🎉 no goals
  -/


theorem mem_factors_iff_dvd {a p : α} (ha0 : a ≠ 0) (hp : Irreducible p) :
    Associates.mk p ∈ factors (Associates.mk a) ↔ p ∣ a := by
  /-
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : UniqueFactorizationMonoid α
    a p : α
    ha0 : Ne a 0
    hp : Irreducible p
    ⊢ Iff (Membership.mem (Associates.mk a).factors (Associates.mk p)) (Dvd.dvd p a)
  -/
  constructor
    /-
      case mp
      α : Type u_1
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : UniqueFactorizationMonoid α
      a p : α
      ha0 : Ne a 0
      hp : Irreducible p
      ⊢ Membership.mem (Associates.mk a).factors (Associates.mk p) → Dvd.dvd p a
    -/
  · rw [← mk_dvd_mk]
    /-
      case mp
      α : Type u_1
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : UniqueFactorizationMonoid α
      a p : α
      ha0 : Ne a 0
      hp : Irreducible p
      ⊢ Membership.mem (Associates.mk a).factors (Associates.mk p) → Dvd.dvd (Associ …
    -/
    apply dvd_of_mem_factors
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u_1
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : UniqueFactorizationMonoid α
      a p : α
      ha0 : Ne a 0
      hp : Irreducible p
      ⊢ Dvd.dvd p a → Membership.mem (Associates.mk a).factors (Associates.mk p)
    -/
  · apply mem_factors_of_dvd ha0 hp
    /-
      🎉 no goals
    -/


open Classical in
theorem exists_prime_dvd_of_not_inf_one {a b : α} (ha : a ≠ 0) (hb : b ≠ 0)
    (h : Associates.mk a ⊓ Associates.mk b ≠ 1) : ∃ p : α, Prime p ∧ p ∣ a ∧ p ∣ b := by
  have hz : factors (Associates.mk a) ⊓ factors (Associates.mk b) ≠ 0 := by
    contrapose! h with hf
    change (factors (Associates.mk a) ⊓ factors (Associates.mk b)).prod = 1
    rw [hf]
    exact Multiset.prod_zero
  /-
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : UniqueFactorizationMonoid α
    a b : α
    ha : Ne a 0
    hb : Ne b 0
    h : Ne (Min.min (Associates.mk a) (Associates.mk b)) 1
    hz : Ne (Min.min (Associates.mk a).factors (Associates.mk b).factors) 0
    ⊢ Exists fun p => And (Prime p) (And (Dvd.dvd p a) (Dvd.dvd p b))
  -/
  rw [factors_mk a ha, factors_mk b hb, ← WithTop.coe_inf] at hz
  /-
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : UniqueFactorizationMonoid α
    a b : α
    ha : Ne a 0
    hb : Ne b 0
    h : Ne (Min.min (Associates.mk a) (Associates.mk b)) 1
    hz : Ne (↑(Min.min (Associates.factors' a) (Associates.factors' b))) 0
    ⊢ Exists fun p => And (Prime p) (And (Dvd.dvd p a) (Dvd.dvd p b))
  -/
  obtain ⟨⟨p0, p0_irr⟩, p0_mem⟩ := Multiset.exists_mem_of_ne_zero ((mt WithTop.coe_eq_coe.mpr) hz)
  /-
    case intro.mk
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : UniqueFactorizationMonoid α
    a b : α
    ha : Ne a 0
    hb : Ne b 0
    h : Ne (Min.min (Associates.mk a) (Associates.mk b)) 1
    hz : Ne (↑(Min.min (Associates.factors' a) (Associates.factors' b))) 0
    p0 : Associates α
    p0_irr : Irreducible p0
    p0_mem : Membership.mem (Min.min (Associates.factors' a) (Associates.factors'  …
    ⊢ Exists fun p => And (Prime p) (And (Dvd.dvd p a) (Dvd.dvd p b))
  -/
  rw [Multiset.inf_eq_inter] at p0_mem
  /-
    case intro.mk
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : UniqueFactorizationMonoid α
    a b : α
    ha : Ne a 0
    hb : Ne b 0
    h : Ne (Min.min (Associates.mk a) (Associates.mk b)) 1
    hz : Ne (↑(Min.min (Associates.factors' a) (Associates.factors' b))) 0
    p0 : Associates α
    p0_irr : Irreducible p0
    p0_mem : Membership.mem (Inter.inter (Associates.factors' a) (Associates.facto …
    ⊢ Exists fun p => And (Prime p) (And (Dvd.dvd p a) (Dvd.dvd p b))
  -/
  obtain ⟨p, rfl⟩ : ∃ p, Associates.mk p = p0 := Quot.exists_rep p0
  /-
    case intro.mk.intro
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : UniqueFactorizationMonoid α
    a b : α
    ha : Ne a 0
    hb : Ne b 0
    h : Ne (Min.min (Associates.mk a) (Associates.mk b)) 1
    hz : Ne (↑(Min.min (Associates.factors' a) (Associates.factors' b))) 0
    p : α
    p0_irr : Irreducible (Associates.mk p)
    p0_mem : Membership.mem (Inter.inter (Associates.factors' a) (Associates.facto …
    ⊢ Exists fun p => And (Prime p) (And (Dvd.dvd p a) (Dvd.dvd p b))
  -/
  refine ⟨p, ?_, ?_, ?_⟩
    /-
      case intro.mk.intro.refine_1
      α : Type u_1
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : UniqueFactorizationMonoid α
      a b : α
      ha : Ne a 0
      hb : Ne b 0
      h : Ne (Min.min (Associates.mk a) (Associates.mk b)) 1
      hz : Ne (↑(Min.min (Associates.factors' a) (Associates.factors' b))) 0
      p : α
      p0_irr : Irreducible (Associates.mk p)
      p0_mem : Membership.mem (Inter.inter (Associates.factors' a) (Associates.facto …
      ⊢ Prime p
    -/
  · rw [← UniqueFactorizationMonoid.irreducible_iff_prime, ← irreducible_mk]
    /-
      case intro.mk.intro.refine_1
      α : Type u_1
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : UniqueFactorizationMonoid α
      a b : α
      ha : Ne a 0
      hb : Ne b 0
      h : Ne (Min.min (Associates.mk a) (Associates.mk b)) 1
      hz : Ne (↑(Min.min (Associates.factors' a) (Associates.factors' b))) 0
      p : α
      p0_irr : Irreducible (Associates.mk p)
      p0_mem : Membership.mem (Inter.inter (Associates.factors' a) (Associates.facto …
      ⊢ Irreducible (Associates.mk p)
    -/
    exact p0_irr
    /-
      🎉 no goals
    -/
    /-
      case intro.mk.intro.refine_2
      α : Type u_1
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : UniqueFactorizationMonoid α
      a b : α
      ha : Ne a 0
      hb : Ne b 0
      h : Ne (Min.min (Associates.mk a) (Associates.mk b)) 1
      hz : Ne (↑(Min.min (Associates.factors' a) (Associates.factors' b))) 0
      p : α
      p0_irr : Irreducible (Associates.mk p)
      p0_mem : Membership.mem (Inter.inter (Associates.factors' a) (Associates.facto …
      ⊢ Dvd.dvd p a
    -/
  · apply dvd_of_mk_le_mk
    /-
      case intro.mk.intro.refine_2.a
      α : Type u_1
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : UniqueFactorizationMonoid α
      a b : α
      ha : Ne a 0
      hb : Ne b 0
      h : Ne (Min.min (Associates.mk a) (Associates.mk b)) 1
      hz : Ne (↑(Min.min (Associates.factors' a) (Associates.factors' b))) 0
      p : α
      p0_irr : Irreducible (Associates.mk p)
      p0_mem : Membership.mem (Inter.inter (Associates.factors' a) (Associates.facto …
      ⊢ LE.le (Associates.mk p) (Associates.mk a)
    -/
    apply dvd_of_mem_factors' (Multiset.mem_inter.mp p0_mem).left
    /-
      α : Type u_1
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : UniqueFactorizationMonoid α
      a b : α
      ha : Ne a 0
      hb : Ne b 0
      h : Ne (Min.min (Associates.mk a) (Associates.mk b)) 1
      hz : Ne (↑(Min.min (Associates.factors' a) (Associates.factors' b))) 0
      p : α
      p0_irr : Irreducible (Associates.mk p)
      p0_mem : Membership.mem (Inter.inter (Associates.factors' a) (Associates.facto …
      ⊢ Ne a 0
    -/
    apply ha
    /-
      🎉 no goals
    -/
    /-
      case intro.mk.intro.refine_3
      α : Type u_1
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : UniqueFactorizationMonoid α
      a b : α
      ha : Ne a 0
      hb : Ne b 0
      h : Ne (Min.min (Associates.mk a) (Associates.mk b)) 1
      hz : Ne (↑(Min.min (Associates.factors' a) (Associates.factors' b))) 0
      p : α
      p0_irr : Irreducible (Associates.mk p)
      p0_mem : Membership.mem (Inter.inter (Associates.factors' a) (Associates.facto …
      ⊢ Dvd.dvd p b
    -/
  · apply dvd_of_mk_le_mk
    /-
      case intro.mk.intro.refine_3.a
      α : Type u_1
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : UniqueFactorizationMonoid α
      a b : α
      ha : Ne a 0
      hb : Ne b 0
      h : Ne (Min.min (Associates.mk a) (Associates.mk b)) 1
      hz : Ne (↑(Min.min (Associates.factors' a) (Associates.factors' b))) 0
      p : α
      p0_irr : Irreducible (Associates.mk p)
      p0_mem : Membership.mem (Inter.inter (Associates.factors' a) (Associates.facto …
      ⊢ LE.le (Associates.mk p) (Associates.mk b)
    -/
    apply dvd_of_mem_factors' (Multiset.mem_inter.mp p0_mem).right
    /-
      α : Type u_1
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : UniqueFactorizationMonoid α
      a b : α
      ha : Ne a 0
      hb : Ne b 0
      h : Ne (Min.min (Associates.mk a) (Associates.mk b)) 1
      hz : Ne (↑(Min.min (Associates.factors' a) (Associates.factors' b))) 0
      p : α
      p0_irr : Irreducible (Associates.mk p)
      p0_mem : Membership.mem (Inter.inter (Associates.factors' a) (Associates.facto …
      ⊢ Ne b 0
    -/
    apply hb
    /-
      🎉 no goals
    -/


theorem coprime_iff_inf_one {a b : α} (ha0 : a ≠ 0) (hb0 : b ≠ 0) :
    Associates.mk a ⊓ Associates.mk b = 1 ↔ ∀ {d : α}, d ∣ a → d ∣ b → ¬Prime d := by
  /-
    α : Type u_1
    inst✝¹ : CancelCommMonoidWithZero α
    inst✝ : UniqueFactorizationMonoid α
    a b : α
    ha0 : Ne a 0
    hb0 : Ne b 0
    ⊢ Iff (Eq (Min.min (Associates.mk a) (Associates.mk b)) 1) (∀ {d : α}, Dvd.dvd …
  -/
  constructor
    /-
      case mp
      α : Type u_1
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : UniqueFactorizationMonoid α
      a b : α
      ha0 : Ne a 0
      hb0 : Ne b 0
      ⊢ Eq (Min.min (Associates.mk a) (Associates.mk b)) 1 → ∀ {d : α}, Dvd.dvd d a  …
    -/
  · intro hg p ha hb hp
    /-
      case mp
      α : Type u_1
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : UniqueFactorizationMonoid α
      a b : α
      ha0 : Ne a 0
      hb0 : Ne b 0
      hg : Eq (Min.min (Associates.mk a) (Associates.mk b)) 1
      p : α
      ha : Dvd.dvd p a
      hb : Dvd.dvd p b
      hp : Prime p
      ⊢ False
    -/
    refine (Associates.prime_mk.mpr hp).not_unit (isUnit_of_dvd_one ?_)
    /-
      case mp
      α : Type u_1
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : UniqueFactorizationMonoid α
      a b : α
      ha0 : Ne a 0
      hb0 : Ne b 0
      hg : Eq (Min.min (Associates.mk a) (Associates.mk b)) 1
      p : α
      ha : Dvd.dvd p a
      hb : Dvd.dvd p b
      hp : Prime p
      ⊢ Dvd.dvd (Associates.mk p) 1
    -/
    rw [← hg]
    /-
      case mp
      α : Type u_1
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : UniqueFactorizationMonoid α
      a b : α
      ha0 : Ne a 0
      hb0 : Ne b 0
      hg : Eq (Min.min (Associates.mk a) (Associates.mk b)) 1
      p : α
      ha : Dvd.dvd p a
      hb : Dvd.dvd p b
      hp : Prime p
      ⊢ Dvd.dvd (Associates.mk p) (Min.min (Associates.mk a) (Associates.mk b))
    -/
    exact le_inf (mk_le_mk_of_dvd ha) (mk_le_mk_of_dvd hb)
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u_1
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : UniqueFactorizationMonoid α
      a b : α
      ha0 : Ne a 0
      hb0 : Ne b 0
      ⊢ (∀ {d : α}, Dvd.dvd d a → Dvd.dvd d b → Not (Prime d)) → Eq (Min.min (Associ …
    -/
  · contrapose
    /-
      case mpr
      α : Type u_1
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : UniqueFactorizationMonoid α
      a b : α
      ha0 : Ne a 0
      hb0 : Ne b 0
      ⊢ Not (Eq (Min.min (Associates.mk a) (Associates.mk b)) 1) → Not (∀ {d : α}, D …
    -/
    intro hg hc
    /-
      case mpr
      α : Type u_1
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : UniqueFactorizationMonoid α
      a b : α
      ha0 : Ne a 0
      hb0 : Ne b 0
      hg : Not (Eq (Min.min (Associates.mk a) (Associates.mk b)) 1)
      hc : ∀ {d : α}, Dvd.dvd d a → Dvd.dvd d b → Not (Prime d)
      ⊢ False
    -/
    obtain ⟨p, hp, hpa, hpb⟩ := exists_prime_dvd_of_not_inf_one ha0 hb0 hg
    /-
      case mpr.intro.intro.intro
      α : Type u_1
      inst✝¹ : CancelCommMonoidWithZero α
      inst✝ : UniqueFactorizationMonoid α
      a b : α
      ha0 : Ne a 0
      hb0 : Ne b 0
      hg : Not (Eq (Min.min (Associates.mk a) (Associates.mk b)) 1)
      hc : ∀ {d : α}, Dvd.dvd d a → Dvd.dvd d b → Not (Prime d)
      p : α
      hp : Prime p
      hpa : Dvd.dvd p a
      hpb : Dvd.dvd p b
      ⊢ False
    -/
    exact hc hpa hpb hp
    /-
      🎉 no goals
    -/


theorem factors_self [Nontrivial α] {p : Associates α} (hp : Irreducible p) :
    p.factors = WithTop.some {⟨p, hp⟩} :=
  eq_of_prod_eq_prod
        /-
          α : Type u_1
          inst✝² : CancelCommMonoidWithZero α
          inst✝¹ : UniqueFactorizationMonoid α
          inst✝ : Nontrivial α
          p : Associates α
          hp : Irreducible p
          ⊢ Eq p.factors.prod (Associates.FactorSet.prod ↑(Singleton.singleton ⟨p, hp⟩))
        -/
    (by rw [factors_prod, FactorSet.prod.eq_def]; dsimp; rw [prod_singleton])
                                                         /-
                                                           🎉 no goals
                                                         -/


theorem factors_prime_pow [Nontrivial α] {p : Associates α} (hp : Irreducible p) (k : ℕ) :
    factors (p ^ k) = WithTop.some (Multiset.replicate k ⟨p, hp⟩) :=
  eq_of_prod_eq_prod
    (by
      /-
        α : Type u_1
        inst✝² : CancelCommMonoidWithZero α
        inst✝¹ : UniqueFactorizationMonoid α
        inst✝ : Nontrivial α
        p : Associates α
        hp : Irreducible p
        k : Nat
        ⊢ Eq (HPow.hPow p k).factors.prod (Associates.FactorSet.prod ↑(Multiset.replic …
      -/
      rw [Associates.factors_prod, FactorSet.prod.eq_def]
      /-
        α : Type u_1
        inst✝² : CancelCommMonoidWithZero α
        inst✝¹ : UniqueFactorizationMonoid α
        inst✝ : Nontrivial α
        p : Associates α
        hp : Irreducible p
        k : Nat
        ⊢ Eq (HPow.hPow p k) (Associates.FactorSet.prod.match_1 (fun x => Associates α …
      -/
      dsimp; rw [Multiset.map_replicate, Multiset.prod_replicate, Subtype.coe_mk])
             /-
               🎉 no goals
             -/


theorem prime_pow_le_iff_le_bcount [DecidableEq (Associates α)] {m p : Associates α}
    (h₁ : m ≠ 0) (h₂ : Irreducible p) {k : ℕ} : p ^ k ≤ m ↔ k ≤ bcount ⟨p, h₂⟩ m.factors := by
  /-
    α : Type u_1
    inst✝² : CancelCommMonoidWithZero α
    inst✝¹ : UniqueFactorizationMonoid α
    inst✝ : DecidableEq (Associates α)
    m p : Associates α
    h₁ : Ne m 0
    h₂ : Irreducible p
    k : Nat
    ⊢ Iff (LE.le (HPow.hPow p k) m) (LE.le k (Associates.bcount ⟨p, h₂⟩ m.factors))
  -/
  rcases Associates.exists_non_zero_rep h₁ with ⟨m, hm, rfl⟩
  /-
    case intro.intro
    α : Type u_1
    inst✝² : CancelCommMonoidWithZero α
    inst✝¹ : UniqueFactorizationMonoid α
    inst✝ : DecidableEq (Associates α)
    p : Associates α
    h₂ : Irreducible p
    k : Nat
    m : α
    hm : Ne m 0
    h₁ : Ne (Associates.mk m) 0
    ⊢ Iff (LE.le (HPow.hPow p k) (Associates.mk m)) (LE.le k (Associates.bcount ⟨p …
  -/
  have := nontrivial_of_ne _ _ hm
  rw [bcount.eq_def, factors_mk, Multiset.le_count_iff_replicate_le, ← factors_le,
                                                           /-
                                                             case intro.intro.h
                                                             α : Type u_1
                                                             inst✝² : CancelCommMonoidWithZero α
                                                             inst✝¹ : UniqueFactorizationMonoid α
                                                             inst✝ : DecidableEq (Associates α)
                                                             p : Associates α
                                                             h₂ : Irreducible p
                                                             k : Nat
                                                             m : α
                                                             hm : Ne m 0
                                                             h₁ : Ne (Associates.mk m) 0
                                                             this : Nontrivial α
                                                             ⊢ Ne m 0
                                                           -/
                                                           /-
                                                             🎉 no goals
                                                           -/
    factors_prime_pow, factors_mk, WithTop.coe_le_coe] <;> assumption
                                                           /-
                                                             🎉 no goals
                                                           -/


@[simp]
theorem factors_one [Nontrivial α] : factors (1 : Associates α) = 0 := by
  /-
    α : Type u_1
    inst✝² : CancelCommMonoidWithZero α
    inst✝¹ : UniqueFactorizationMonoid α
    inst✝ : Nontrivial α
    ⊢ Eq (Associates.factors 1) 0
  -/
  apply eq_of_prod_eq_prod
  /-
    case h
    α : Type u_1
    inst✝² : CancelCommMonoidWithZero α
    inst✝¹ : UniqueFactorizationMonoid α
    inst✝ : Nontrivial α
    ⊢ Eq (Associates.factors 1).prod (Associates.FactorSet.prod 0)
  -/
  rw [Associates.factors_prod]
  /-
    case h
    α : Type u_1
    inst✝² : CancelCommMonoidWithZero α
    inst✝¹ : UniqueFactorizationMonoid α
    inst✝ : Nontrivial α
    ⊢ Eq 1 (Associates.FactorSet.prod 0)
  -/
  exact Multiset.prod_zero
  /-
    🎉 no goals
  -/


@[simp]
theorem pow_factors [Nontrivial α] {a : Associates α} {k : ℕ} :
    (a ^ k).factors = k • a.factors := by
  /-
    α : Type u_1
    inst✝² : CancelCommMonoidWithZero α
    inst✝¹ : UniqueFactorizationMonoid α
    inst✝ : Nontrivial α
    a : Associates α
    k : Nat
    ⊢ Eq (HPow.hPow a k).factors (HSMul.hSMul k a.factors)
  -/
  induction' k with n h
    /-
      case zero
      α : Type u_1
      inst✝² : CancelCommMonoidWithZero α
      inst✝¹ : UniqueFactorizationMonoid α
      inst✝ : Nontrivial α
      a : Associates α
      ⊢ Eq (HPow.hPow a 0).factors (HSMul.hSMul 0 a.factors)
    -/
  · rw [zero_nsmul, pow_zero]
    /-
      case zero
      α : Type u_1
      inst✝² : CancelCommMonoidWithZero α
      inst✝¹ : UniqueFactorizationMonoid α
      inst✝ : Nontrivial α
      a : Associates α
      ⊢ Eq (Associates.factors 1) 0
    -/
    exact factors_one
    /-
      🎉 no goals
    -/
    /-
      case succ
      α : Type u_1
      inst✝² : CancelCommMonoidWithZero α
      inst✝¹ : UniqueFactorizationMonoid α
      inst✝ : Nontrivial α
      a : Associates α
      n : Nat
      h : Eq (HPow.hPow a n).factors (HSMul.hSMul n a.factors)
      ⊢ Eq (HPow.hPow a (HAdd.hAdd n 1)).factors (HSMul.hSMul (HAdd.hAdd n 1) a.fact …
    -/
  · rw [pow_succ, succ_nsmul, factors_mul, h]
    /-
      🎉 no goals
    -/


theorem prime_pow_dvd_iff_le {m p : Associates α} (h₁ : m ≠ 0) (h₂ : Irreducible p) {k : ℕ} :
    p ^ k ≤ m ↔ k ≤ count p m.factors := by
  /-
    α : Type u_1
    inst✝³ : CancelCommMonoidWithZero α
    inst✝² : UniqueFactorizationMonoid α
    inst✝¹ : DecidableEq (Associates α)
    inst✝ : (p : Associates α) → Decidable (Irreducible p)
    m p : Associates α
    h₁ : Ne m 0
    h₂ : Irreducible p
    k : Nat
    ⊢ Iff (LE.le (HPow.hPow p k) m) (LE.le k (p.count m.factors))
  -/
  rw [count, dif_pos h₂, prime_pow_le_iff_le_bcount h₁]
  /-
    🎉 no goals
  -/


theorem le_of_count_ne_zero {m p : Associates α} (h0 : m ≠ 0) (hp : Irreducible p) :
    count p m.factors ≠ 0 → p ≤ m := by
  /-
    α : Type u_1
    inst✝³ : CancelCommMonoidWithZero α
    inst✝² : UniqueFactorizationMonoid α
    inst✝¹ : DecidableEq (Associates α)
    inst✝ : (p : Associates α) → Decidable (Irreducible p)
    m p : Associates α
    h0 : Ne m 0
    hp : Irreducible p
    ⊢ Ne (p.count m.factors) 0 → LE.le p m
  -/
  nontriviality α
  /-
    α : Type u_1
    inst✝³ : CancelCommMonoidWithZero α
    inst✝² : UniqueFactorizationMonoid α
    inst✝¹ : DecidableEq (Associates α)
    inst✝ : (p : Associates α) → Decidable (Irreducible p)
    m p : Associates α
    h0 : Ne m 0
    hp : Irreducible p
    a✝ : Nontrivial α
    ⊢ Ne (p.count m.factors) 0 → LE.le p m
  -/
  rw [← pos_iff_ne_zero]
  /-
    α : Type u_1
    inst✝³ : CancelCommMonoidWithZero α
    inst✝² : UniqueFactorizationMonoid α
    inst✝¹ : DecidableEq (Associates α)
    inst✝ : (p : Associates α) → Decidable (Irreducible p)
    m p : Associates α
    h0 : Ne m 0
    hp : Irreducible p
    a✝ : Nontrivial α
    ⊢ LT.lt 0 (p.count m.factors) → LE.le p m
  -/
  intro h
  /-
    α : Type u_1
    inst✝³ : CancelCommMonoidWithZero α
    inst✝² : UniqueFactorizationMonoid α
    inst✝¹ : DecidableEq (Associates α)
    inst✝ : (p : Associates α) → Decidable (Irreducible p)
    m p : Associates α
    h0 : Ne m 0
    hp : Irreducible p
    a✝ : Nontrivial α
    h : LT.lt 0 (p.count m.factors)
    ⊢ LE.le p m
  -/
  rw [← pow_one p]
  /-
    α : Type u_1
    inst✝³ : CancelCommMonoidWithZero α
    inst✝² : UniqueFactorizationMonoid α
    inst✝¹ : DecidableEq (Associates α)
    inst✝ : (p : Associates α) → Decidable (Irreducible p)
    m p : Associates α
    h0 : Ne m 0
    hp : Irreducible p
    a✝ : Nontrivial α
    h : LT.lt 0 (p.count m.factors)
    ⊢ LE.le (HPow.hPow p 1) m
  -/
  apply (prime_pow_dvd_iff_le h0 hp).2
  /-
    α : Type u_1
    inst✝³ : CancelCommMonoidWithZero α
    inst✝² : UniqueFactorizationMonoid α
    inst✝¹ : DecidableEq (Associates α)
    inst✝ : (p : Associates α) → Decidable (Irreducible p)
    m p : Associates α
    h0 : Ne m 0
    hp : Irreducible p
    a✝ : Nontrivial α
    h : LT.lt 0 (p.count m.factors)
    ⊢ LE.le 1 (p.count m.factors)
  -/
  simpa only
  /-
    🎉 no goals
  -/


theorem count_ne_zero_iff_dvd {a p : α} (ha0 : a ≠ 0) (hp : Irreducible p) :
    (Associates.mk p).count (Associates.mk a).factors ≠ 0 ↔ p ∣ a := by
  /-
    α : Type u_1
    inst✝³ : CancelCommMonoidWithZero α
    inst✝² : UniqueFactorizationMonoid α
    inst✝¹ : DecidableEq (Associates α)
    inst✝ : (p : Associates α) → Decidable (Irreducible p)
    a p : α
    ha0 : Ne a 0
    hp : Irreducible p
    ⊢ Iff (Ne ((Associates.mk p).count (Associates.mk a).factors) 0) (Dvd.dvd p a)
  -/
  nontriviality α
  /-
    α : Type u_1
    inst✝⁴ : CancelCommMonoidWithZero α
    inst✝³ : UniqueFactorizationMonoid α
    inst✝² : DecidableEq (Associates α)
    inst✝¹ : (p : Associates α) → Decidable (Irreducible p)
    a p : α
    ha0 : Ne a 0
    hp : Irreducible p
    inst✝ : Nontrivial α
    ⊢ Iff (Ne ((Associates.mk p).count (Associates.mk a).factors) 0) (Dvd.dvd p a)
  -/
  rw [← Associates.mk_le_mk_iff_dvd]
  refine
    ⟨fun h =>
      Associates.le_of_count_ne_zero (Associates.mk_ne_zero.mpr ha0)
        (Associates.irreducible_mk.mpr hp) h,
      fun h => ?_⟩
  rw [← pow_one (Associates.mk p),
    Associates.prime_pow_dvd_iff_le (Associates.mk_ne_zero.mpr ha0)
      (Associates.irreducible_mk.mpr hp)] at h
  /-
    α : Type u_1
    inst✝⁴ : CancelCommMonoidWithZero α
    inst✝³ : UniqueFactorizationMonoid α
    inst✝² : DecidableEq (Associates α)
    inst✝¹ : (p : Associates α) → Decidable (Irreducible p)
    a p : α
    ha0 : Ne a 0
    hp : Irreducible p
    inst✝ : Nontrivial α
    h : LE.le 1 ((Associates.mk p).count (Associates.mk a).factors)
    ⊢ Ne ((Associates.mk p).count (Associates.mk a).factors) 0
  -/
  exact (zero_lt_one.trans_le h).ne'
  /-
    🎉 no goals
  -/


theorem count_self [Nontrivial α] {p : Associates α}
    (hp : Irreducible p) : p.count p.factors = 1 := by
  /-
    α : Type u_1
    inst✝⁴ : CancelCommMonoidWithZero α
    inst✝³ : UniqueFactorizationMonoid α
    inst✝² : DecidableEq (Associates α)
    inst✝¹ : (p : Associates α) → Decidable (Irreducible p)
    inst✝ : Nontrivial α
    p : Associates α
    hp : Irreducible p
    ⊢ Eq (p.count p.factors) 1
  -/
  simp [factors_self hp, Associates.count_some hp]
  /-
    🎉 no goals
  -/


theorem count_eq_zero_of_ne {p q : Associates α} (hp : Irreducible p)
    (hq : Irreducible q) (h : p ≠ q) : p.count q.factors = 0 :=
  not_ne_iff.mp fun h' ↦ h <| associated_iff_eq.mp <| hp.associated_of_dvd hq <|
    le_of_count_ne_zero hq.ne_zero hp h'


theorem count_mul {a : Associates α} (ha : a ≠ 0) {b : Associates α}
    (hb : b ≠ 0) {p : Associates α} (hp : Irreducible p) :
    count p (factors (a * b)) = count p a.factors + count p b.factors := by
  /-
    α : Type u_1
    inst✝³ : CancelCommMonoidWithZero α
    inst✝² : UniqueFactorizationMonoid α
    inst✝¹ : DecidableEq (Associates α)
    inst✝ : (p : Associates α) → Decidable (Irreducible p)
    a : Associates α
    ha : Ne a 0
    b : Associates α
    hb : Ne b 0
    p : Associates α
    hp : Irreducible p
    ⊢ Eq (p.count (HMul.hMul a b).factors) (HAdd.hAdd (p.count a.factors) (p.count …
  -/
  obtain ⟨a0, nza, rfl⟩ := exists_non_zero_rep ha
  /-
    case intro.intro
    α : Type u_1
    inst✝³ : CancelCommMonoidWithZero α
    inst✝² : UniqueFactorizationMonoid α
    inst✝¹ : DecidableEq (Associates α)
    inst✝ : (p : Associates α) → Decidable (Irreducible p)
    b : Associates α
    hb : Ne b 0
    p : Associates α
    hp : Irreducible p
    a0 : α
    nza : Ne a0 0
    ha : Ne (Associates.mk a0) 0
    ⊢ Eq (p.count (HMul.hMul (Associates.mk a0) b).factors) (HAdd.hAdd (p.count (A …
  -/
  obtain ⟨b0, nzb, rfl⟩ := exists_non_zero_rep hb
  rw [factors_mul, factors_mk a0 nza, factors_mk b0 nzb, ← FactorSet.coe_add, count_some hp,
    Multiset.count_add, count_some hp, count_some hp]


theorem count_of_coprime {a : Associates α} (ha : a ≠ 0)
    {b : Associates α} (hb : b ≠ 0) (hab : ∀ d, d ∣ a → d ∣ b → ¬Prime d) {p : Associates α}
    (hp : Irreducible p) : count p a.factors = 0 ∨ count p b.factors = 0 := by
  /-
    α : Type u_1
    inst✝³ : CancelCommMonoidWithZero α
    inst✝² : UniqueFactorizationMonoid α
    inst✝¹ : DecidableEq (Associates α)
    inst✝ : (p : Associates α) → Decidable (Irreducible p)
    a : Associates α
    ha : Ne a 0
    b : Associates α
    hb : Ne b 0
    hab : ∀ (d : Associates α), Dvd.dvd d a → Dvd.dvd d b → Not (Prime d)
    p : Associates α
    hp : Irreducible p
    ⊢ Or (Eq (p.count a.factors) 0) (Eq (p.count b.factors) 0)
  -/
  rw [or_iff_not_imp_left, ← Ne]
  /-
    α : Type u_1
    inst✝³ : CancelCommMonoidWithZero α
    inst✝² : UniqueFactorizationMonoid α
    inst✝¹ : DecidableEq (Associates α)
    inst✝ : (p : Associates α) → Decidable (Irreducible p)
    a : Associates α
    ha : Ne a 0
    b : Associates α
    hb : Ne b 0
    hab : ∀ (d : Associates α), Dvd.dvd d a → Dvd.dvd d b → Not (Prime d)
    p : Associates α
    hp : Irreducible p
    ⊢ Ne (p.count a.factors) 0 → Eq (p.count b.factors) 0
  -/
  intro hca
  /-
    α : Type u_1
    inst✝³ : CancelCommMonoidWithZero α
    inst✝² : UniqueFactorizationMonoid α
    inst✝¹ : DecidableEq (Associates α)
    inst✝ : (p : Associates α) → Decidable (Irreducible p)
    a : Associates α
    ha : Ne a 0
    b : Associates α
    hb : Ne b 0
    hab : ∀ (d : Associates α), Dvd.dvd d a → Dvd.dvd d b → Not (Prime d)
    p : Associates α
    hp : Irreducible p
    hca : Ne (p.count a.factors) 0
    ⊢ Eq (p.count b.factors) 0
  -/
  contrapose! hab with hcb
  exact ⟨p, le_of_count_ne_zero ha hp hca, le_of_count_ne_zero hb hp hcb,
    UniqueFactorizationMonoid.irreducible_iff_prime.mp hp⟩


theorem count_mul_of_coprime {a : Associates α} {b : Associates α}
    (hb : b ≠ 0) {p : Associates α} (hp : Irreducible p) (hab : ∀ d, d ∣ a → d ∣ b → ¬Prime d) :
    count p a.factors = 0 ∨ count p a.factors = count p (a * b).factors := by
  /-
    α : Type u_1
    inst✝³ : CancelCommMonoidWithZero α
    inst✝² : UniqueFactorizationMonoid α
    inst✝¹ : DecidableEq (Associates α)
    inst✝ : (p : Associates α) → Decidable (Irreducible p)
    a b : Associates α
    hb : Ne b 0
    p : Associates α
    hp : Irreducible p
    hab : ∀ (d : Associates α), Dvd.dvd d a → Dvd.dvd d b → Not (Prime d)
    ⊢ Or (Eq (p.count a.factors) 0) (Eq (p.count a.factors) (p.count (HMul.hMul a  …
  -/
  by_cases ha : a = 0
    /-
      case pos
      α : Type u_1
      inst✝³ : CancelCommMonoidWithZero α
      inst✝² : UniqueFactorizationMonoid α
      inst✝¹ : DecidableEq (Associates α)
      inst✝ : (p : Associates α) → Decidable (Irreducible p)
      a b : Associates α
      hb : Ne b 0
      p : Associates α
      hp : Irreducible p
      hab : ∀ (d : Associates α), Dvd.dvd d a → Dvd.dvd d b → Not (Prime d)
      ha : Eq a 0
      ⊢ Or (Eq (p.count a.factors) 0) (Eq (p.count a.factors) (p.count (HMul.hMul a  …
    -/
  · simp [ha]
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    inst✝³ : CancelCommMonoidWithZero α
    inst✝² : UniqueFactorizationMonoid α
    inst✝¹ : DecidableEq (Associates α)
    inst✝ : (p : Associates α) → Decidable (Irreducible p)
    a b : Associates α
    hb : Ne b 0
    p : Associates α
    hp : Irreducible p
    hab : ∀ (d : Associates α), Dvd.dvd d a → Dvd.dvd d b → Not (Prime d)
    ha : Not (Eq a 0)
    ⊢ Or (Eq (p.count a.factors) 0) (Eq (p.count a.factors) (p.count (HMul.hMul a  …
  -/
  cases' count_of_coprime ha hb hab hp with hz hb0; · tauto
                                                      /-
                                                        🎉 no goals
                                                      -/
  /-
    case neg.inr
    α : Type u_1
    inst✝³ : CancelCommMonoidWithZero α
    inst✝² : UniqueFactorizationMonoid α
    inst✝¹ : DecidableEq (Associates α)
    inst✝ : (p : Associates α) → Decidable (Irreducible p)
    a b : Associates α
    hb : Ne b 0
    p : Associates α
    hp : Irreducible p
    hab : ∀ (d : Associates α), Dvd.dvd d a → Dvd.dvd d b → Not (Prime d)
    ha : Not (Eq a 0)
    hb0 : Eq (p.count b.factors) 0
    ⊢ Or (Eq (p.count a.factors) 0) (Eq (p.count a.factors) (p.count (HMul.hMul a  …
  -/
  apply Or.intro_right
  /-
    case neg.inr.h
    α : Type u_1
    inst✝³ : CancelCommMonoidWithZero α
    inst✝² : UniqueFactorizationMonoid α
    inst✝¹ : DecidableEq (Associates α)
    inst✝ : (p : Associates α) → Decidable (Irreducible p)
    a b : Associates α
    hb : Ne b 0
    p : Associates α
    hp : Irreducible p
    hab : ∀ (d : Associates α), Dvd.dvd d a → Dvd.dvd d b → Not (Prime d)
    ha : Not (Eq a 0)
    hb0 : Eq (p.count b.factors) 0
    ⊢ Eq (p.count a.factors) (p.count (HMul.hMul a b).factors)
  -/
  rw [count_mul ha hb hp, hb0, add_zero]
  /-
    🎉 no goals
  -/


theorem count_mul_of_coprime' {a b : Associates α} {p : Associates α}
    (hp : Irreducible p) (hab : ∀ d, d ∣ a → d ∣ b → ¬Prime d) :
    count p (a * b).factors = count p a.factors ∨ count p (a * b).factors = count p b.factors := by
  /-
    α : Type u_1
    inst✝³ : CancelCommMonoidWithZero α
    inst✝² : UniqueFactorizationMonoid α
    inst✝¹ : DecidableEq (Associates α)
    inst✝ : (p : Associates α) → Decidable (Irreducible p)
    a b p : Associates α
    hp : Irreducible p
    hab : ∀ (d : Associates α), Dvd.dvd d a → Dvd.dvd d b → Not (Prime d)
    ⊢ Or (Eq (p.count (HMul.hMul a b).factors) (p.count a.factors)) (Eq (p.count ( …
  -/
  by_cases ha : a = 0
    /-
      case pos
      α : Type u_1
      inst✝³ : CancelCommMonoidWithZero α
      inst✝² : UniqueFactorizationMonoid α
      inst✝¹ : DecidableEq (Associates α)
      inst✝ : (p : Associates α) → Decidable (Irreducible p)
      a b p : Associates α
      hp : Irreducible p
      hab : ∀ (d : Associates α), Dvd.dvd d a → Dvd.dvd d b → Not (Prime d)
      ha : Eq a 0
      ⊢ Or (Eq (p.count (HMul.hMul a b).factors) (p.count a.factors)) (Eq (p.count ( …
    -/
  · simp [ha]
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    inst✝³ : CancelCommMonoidWithZero α
    inst✝² : UniqueFactorizationMonoid α
    inst✝¹ : DecidableEq (Associates α)
    inst✝ : (p : Associates α) → Decidable (Irreducible p)
    a b p : Associates α
    hp : Irreducible p
    hab : ∀ (d : Associates α), Dvd.dvd d a → Dvd.dvd d b → Not (Prime d)
    ha : Not (Eq a 0)
    ⊢ Or (Eq (p.count (HMul.hMul a b).factors) (p.count a.factors)) (Eq (p.count ( …
  -/
  by_cases hb : b = 0
    /-
      case pos
      α : Type u_1
      inst✝³ : CancelCommMonoidWithZero α
      inst✝² : UniqueFactorizationMonoid α
      inst✝¹ : DecidableEq (Associates α)
      inst✝ : (p : Associates α) → Decidable (Irreducible p)
      a b p : Associates α
      hp : Irreducible p
      hab : ∀ (d : Associates α), Dvd.dvd d a → Dvd.dvd d b → Not (Prime d)
      ha : Not (Eq a 0)
      hb : Eq b 0
      ⊢ Or (Eq (p.count (HMul.hMul a b).factors) (p.count a.factors)) (Eq (p.count ( …
    -/
  · simp [hb]
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    inst✝³ : CancelCommMonoidWithZero α
    inst✝² : UniqueFactorizationMonoid α
    inst✝¹ : DecidableEq (Associates α)
    inst✝ : (p : Associates α) → Decidable (Irreducible p)
    a b p : Associates α
    hp : Irreducible p
    hab : ∀ (d : Associates α), Dvd.dvd d a → Dvd.dvd d b → Not (Prime d)
    ha : Not (Eq a 0)
    hb : Not (Eq b 0)
    ⊢ Or (Eq (p.count (HMul.hMul a b).factors) (p.count a.factors)) (Eq (p.count ( …
  -/
  rw [count_mul ha hb hp]
  /-
    case neg
    α : Type u_1
    inst✝³ : CancelCommMonoidWithZero α
    inst✝² : UniqueFactorizationMonoid α
    inst✝¹ : DecidableEq (Associates α)
    inst✝ : (p : Associates α) → Decidable (Irreducible p)
    a b p : Associates α
    hp : Irreducible p
    hab : ∀ (d : Associates α), Dvd.dvd d a → Dvd.dvd d b → Not (Prime d)
    ha : Not (Eq a 0)
    hb : Not (Eq b 0)
    ⊢ Or (Eq (HAdd.hAdd (p.count a.factors) (p.count b.factors)) (p.count a.factor …
  -/
  cases' count_of_coprime ha hb hab hp with ha0 hb0
    /-
      case neg.inl
      α : Type u_1
      inst✝³ : CancelCommMonoidWithZero α
      inst✝² : UniqueFactorizationMonoid α
      inst✝¹ : DecidableEq (Associates α)
      inst✝ : (p : Associates α) → Decidable (Irreducible p)
      a b p : Associates α
      hp : Irreducible p
      hab : ∀ (d : Associates α), Dvd.dvd d a → Dvd.dvd d b → Not (Prime d)
      ha : Not (Eq a 0)
      hb : Not (Eq b 0)
      ha0 : Eq (p.count a.factors) 0
      ⊢ Or (Eq (HAdd.hAdd (p.count a.factors) (p.count b.factors)) (p.count a.factor …
    -/
  · apply Or.intro_right
    /-
      case neg.inl.h
      α : Type u_1
      inst✝³ : CancelCommMonoidWithZero α
      inst✝² : UniqueFactorizationMonoid α
      inst✝¹ : DecidableEq (Associates α)
      inst✝ : (p : Associates α) → Decidable (Irreducible p)
      a b p : Associates α
      hp : Irreducible p
      hab : ∀ (d : Associates α), Dvd.dvd d a → Dvd.dvd d b → Not (Prime d)
      ha : Not (Eq a 0)
      hb : Not (Eq b 0)
      ha0 : Eq (p.count a.factors) 0
      ⊢ Eq (HAdd.hAdd (p.count a.factors) (p.count b.factors)) (p.count b.factors)
    -/
    rw [ha0, zero_add]
    /-
      🎉 no goals
    -/
    /-
      case neg.inr
      α : Type u_1
      inst✝³ : CancelCommMonoidWithZero α
      inst✝² : UniqueFactorizationMonoid α
      inst✝¹ : DecidableEq (Associates α)
      inst✝ : (p : Associates α) → Decidable (Irreducible p)
      a b p : Associates α
      hp : Irreducible p
      hab : ∀ (d : Associates α), Dvd.dvd d a → Dvd.dvd d b → Not (Prime d)
      ha : Not (Eq a 0)
      hb : Not (Eq b 0)
      hb0 : Eq (p.count b.factors) 0
      ⊢ Or (Eq (HAdd.hAdd (p.count a.factors) (p.count b.factors)) (p.count a.factor …
    -/
  · apply Or.intro_left
    /-
      case neg.inr.h
      α : Type u_1
      inst✝³ : CancelCommMonoidWithZero α
      inst✝² : UniqueFactorizationMonoid α
      inst✝¹ : DecidableEq (Associates α)
      inst✝ : (p : Associates α) → Decidable (Irreducible p)
      a b p : Associates α
      hp : Irreducible p
      hab : ∀ (d : Associates α), Dvd.dvd d a → Dvd.dvd d b → Not (Prime d)
      ha : Not (Eq a 0)
      hb : Not (Eq b 0)
      hb0 : Eq (p.count b.factors) 0
      ⊢ Eq (HAdd.hAdd (p.count a.factors) (p.count b.factors)) (p.count a.factors)
    -/
    rw [hb0, add_zero]
    /-
      🎉 no goals
    -/


theorem dvd_count_of_dvd_count_mul {a b : Associates α} (hb : b ≠ 0)
    {p : Associates α} (hp : Irreducible p) (hab : ∀ d, d ∣ a → d ∣ b → ¬Prime d) {k : ℕ}
    (habk : k ∣ count p (a * b).factors) : k ∣ count p a.factors := by
  /-
    α : Type u_1
    inst✝³ : CancelCommMonoidWithZero α
    inst✝² : UniqueFactorizationMonoid α
    inst✝¹ : DecidableEq (Associates α)
    inst✝ : (p : Associates α) → Decidable (Irreducible p)
    a b : Associates α
    hb : Ne b 0
    p : Associates α
    hp : Irreducible p
    hab : ∀ (d : Associates α), Dvd.dvd d a → Dvd.dvd d b → Not (Prime d)
    k : Nat
    habk : Dvd.dvd k (p.count (HMul.hMul a b).factors)
    ⊢ Dvd.dvd k (p.count a.factors)
  -/
  by_cases ha : a = 0
    /-
      case pos
      α : Type u_1
      inst✝³ : CancelCommMonoidWithZero α
      inst✝² : UniqueFactorizationMonoid α
      inst✝¹ : DecidableEq (Associates α)
      inst✝ : (p : Associates α) → Decidable (Irreducible p)
      a b : Associates α
      hb : Ne b 0
      p : Associates α
      hp : Irreducible p
      hab : ∀ (d : Associates α), Dvd.dvd d a → Dvd.dvd d b → Not (Prime d)
      k : Nat
      habk : Dvd.dvd k (p.count (HMul.hMul a b).factors)
      ha : Eq a 0
      ⊢ Dvd.dvd k (p.count a.factors)
    -/
  · simpa [*] using habk
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    inst✝³ : CancelCommMonoidWithZero α
    inst✝² : UniqueFactorizationMonoid α
    inst✝¹ : DecidableEq (Associates α)
    inst✝ : (p : Associates α) → Decidable (Irreducible p)
    a b : Associates α
    hb : Ne b 0
    p : Associates α
    hp : Irreducible p
    hab : ∀ (d : Associates α), Dvd.dvd d a → Dvd.dvd d b → Not (Prime d)
    k : Nat
    habk : Dvd.dvd k (p.count (HMul.hMul a b).factors)
    ha : Not (Eq a 0)
    ⊢ Dvd.dvd k (p.count a.factors)
  -/
  cases' count_of_coprime ha hb hab hp with hz h
    /-
      case neg.inl
      α : Type u_1
      inst✝³ : CancelCommMonoidWithZero α
      inst✝² : UniqueFactorizationMonoid α
      inst✝¹ : DecidableEq (Associates α)
      inst✝ : (p : Associates α) → Decidable (Irreducible p)
      a b : Associates α
      hb : Ne b 0
      p : Associates α
      hp : Irreducible p
      hab : ∀ (d : Associates α), Dvd.dvd d a → Dvd.dvd d b → Not (Prime d)
      k : Nat
      habk : Dvd.dvd k (p.count (HMul.hMul a b).factors)
      ha : Not (Eq a 0)
      hz : Eq (p.count a.factors) 0
      ⊢ Dvd.dvd k (p.count a.factors)
    -/
  · rw [hz]
    /-
      case neg.inl
      α : Type u_1
      inst✝³ : CancelCommMonoidWithZero α
      inst✝² : UniqueFactorizationMonoid α
      inst✝¹ : DecidableEq (Associates α)
      inst✝ : (p : Associates α) → Decidable (Irreducible p)
      a b : Associates α
      hb : Ne b 0
      p : Associates α
      hp : Irreducible p
      hab : ∀ (d : Associates α), Dvd.dvd d a → Dvd.dvd d b → Not (Prime d)
      k : Nat
      habk : Dvd.dvd k (p.count (HMul.hMul a b).factors)
      ha : Not (Eq a 0)
      hz : Eq (p.count a.factors) 0
      ⊢ Dvd.dvd k 0
    -/
    exact dvd_zero k
    /-
      🎉 no goals
    -/
    /-
      case neg.inr
      α : Type u_1
      inst✝³ : CancelCommMonoidWithZero α
      inst✝² : UniqueFactorizationMonoid α
      inst✝¹ : DecidableEq (Associates α)
      inst✝ : (p : Associates α) → Decidable (Irreducible p)
      a b : Associates α
      hb : Ne b 0
      p : Associates α
      hp : Irreducible p
      hab : ∀ (d : Associates α), Dvd.dvd d a → Dvd.dvd d b → Not (Prime d)
      k : Nat
      habk : Dvd.dvd k (p.count (HMul.hMul a b).factors)
      ha : Not (Eq a 0)
      h : Eq (p.count b.factors) 0
      ⊢ Dvd.dvd k (p.count a.factors)
    -/
  · rw [count_mul ha hb hp, h] at habk
    /-
      case neg.inr
      α : Type u_1
      inst✝³ : CancelCommMonoidWithZero α
      inst✝² : UniqueFactorizationMonoid α
      inst✝¹ : DecidableEq (Associates α)
      inst✝ : (p : Associates α) → Decidable (Irreducible p)
      a b : Associates α
      hb : Ne b 0
      p : Associates α
      hp : Irreducible p
      hab : ∀ (d : Associates α), Dvd.dvd d a → Dvd.dvd d b → Not (Prime d)
      k : Nat
      habk : Dvd.dvd k (HAdd.hAdd (p.count a.factors) 0)
      ha : Not (Eq a 0)
      h : Eq (p.count b.factors) 0
      ⊢ Dvd.dvd k (p.count a.factors)
    -/
    exact habk
    /-
      🎉 no goals
    -/


theorem count_pow [Nontrivial α] {a : Associates α} (ha : a ≠ 0)
    {p : Associates α} (hp : Irreducible p) (k : ℕ) :
    count p (a ^ k).factors = k * count p a.factors := by
  /-
    α : Type u_1
    inst✝⁴ : CancelCommMonoidWithZero α
    inst✝³ : UniqueFactorizationMonoid α
    inst✝² : DecidableEq (Associates α)
    inst✝¹ : (p : Associates α) → Decidable (Irreducible p)
    inst✝ : Nontrivial α
    a : Associates α
    ha : Ne a 0
    p : Associates α
    hp : Irreducible p
    k : Nat
    ⊢ Eq (p.count (HPow.hPow a k).factors) (HMul.hMul k (p.count a.factors))
  -/
  induction' k with n h
    /-
      case zero
      α : Type u_1
      inst✝⁴ : CancelCommMonoidWithZero α
      inst✝³ : UniqueFactorizationMonoid α
      inst✝² : DecidableEq (Associates α)
      inst✝¹ : (p : Associates α) → Decidable (Irreducible p)
      inst✝ : Nontrivial α
      a : Associates α
      ha : Ne a 0
      p : Associates α
      hp : Irreducible p
      ⊢ Eq (p.count (HPow.hPow a 0).factors) (HMul.hMul 0 (p.count a.factors))
    -/
  · rw [pow_zero, factors_one, zero_mul, count_zero hp]
    /-
      🎉 no goals
    -/
    /-
      case succ
      α : Type u_1
      inst✝⁴ : CancelCommMonoidWithZero α
      inst✝³ : UniqueFactorizationMonoid α
      inst✝² : DecidableEq (Associates α)
      inst✝¹ : (p : Associates α) → Decidable (Irreducible p)
      inst✝ : Nontrivial α
      a : Associates α
      ha : Ne a 0
      p : Associates α
      hp : Irreducible p
      n : Nat
      h : Eq (p.count (HPow.hPow a n).factors) (HMul.hMul n (p.count a.factors))
      ⊢ Eq (p.count (HPow.hPow a (HAdd.hAdd n 1)).factors) (HMul.hMul (HAdd.hAdd n 1 …
    -/
  · rw [pow_succ', count_mul ha (pow_ne_zero _ ha) hp, h]
    /-
      case succ
      α : Type u_1
      inst✝⁴ : CancelCommMonoidWithZero α
      inst✝³ : UniqueFactorizationMonoid α
      inst✝² : DecidableEq (Associates α)
      inst✝¹ : (p : Associates α) → Decidable (Irreducible p)
      inst✝ : Nontrivial α
      a : Associates α
      ha : Ne a 0
      p : Associates α
      hp : Irreducible p
      n : Nat
      h : Eq (p.count (HPow.hPow a n).factors) (HMul.hMul n (p.count a.factors))
      ⊢ Eq (HAdd.hAdd (p.count a.factors) (HMul.hMul n (p.count a.factors))) (HMul.h …
    -/
    ring
    /-
      🎉 no goals
    -/


theorem dvd_count_pow [Nontrivial α] {a : Associates α} (ha : a ≠ 0)
    {p : Associates α} (hp : Irreducible p) (k : ℕ) : k ∣ count p (a ^ k).factors := by
  /-
    α : Type u_1
    inst✝⁴ : CancelCommMonoidWithZero α
    inst✝³ : UniqueFactorizationMonoid α
    inst✝² : DecidableEq (Associates α)
    inst✝¹ : (p : Associates α) → Decidable (Irreducible p)
    inst✝ : Nontrivial α
    a : Associates α
    ha : Ne a 0
    p : Associates α
    hp : Irreducible p
    k : Nat
    ⊢ Dvd.dvd k (p.count (HPow.hPow a k).factors)
  -/
  rw [count_pow ha hp]
  /-
    α : Type u_1
    inst✝⁴ : CancelCommMonoidWithZero α
    inst✝³ : UniqueFactorizationMonoid α
    inst✝² : DecidableEq (Associates α)
    inst✝¹ : (p : Associates α) → Decidable (Irreducible p)
    inst✝ : Nontrivial α
    a : Associates α
    ha : Ne a 0
    p : Associates α
    hp : Irreducible p
    k : Nat
    ⊢ Dvd.dvd k (HMul.hMul k (p.count a.factors))
  -/
  apply dvd_mul_right
  /-
    🎉 no goals
  -/


theorem is_pow_of_dvd_count {a : Associates α}
    (ha : a ≠ 0) {k : ℕ} (hk : ∀ p : Associates α, Irreducible p → k ∣ count p a.factors) :
    ∃ b : Associates α, a = b ^ k := by
  /-
    α : Type u_1
    inst✝³ : CancelCommMonoidWithZero α
    inst✝² : UniqueFactorizationMonoid α
    inst✝¹ : DecidableEq (Associates α)
    inst✝ : (p : Associates α) → Decidable (Irreducible p)
    a : Associates α
    ha : Ne a 0
    k : Nat
    hk : ∀ (p : Associates α), Irreducible p → Dvd.dvd k (p.count a.factors)
    ⊢ Exists fun b => Eq a (HPow.hPow b k)
  -/
  nontriviality α
  /-
    α : Type u_1
    inst✝³ : CancelCommMonoidWithZero α
    inst✝² : UniqueFactorizationMonoid α
    inst✝¹ : DecidableEq (Associates α)
    inst✝ : (p : Associates α) → Decidable (Irreducible p)
    a : Associates α
    ha : Ne a 0
    k : Nat
    hk : ∀ (p : Associates α), Irreducible p → Dvd.dvd k (p.count a.factors)
    a✝ : Nontrivial α
    ⊢ Exists fun b => Eq a (HPow.hPow b k)
  -/
  obtain ⟨a0, hz, rfl⟩ := exists_non_zero_rep ha
  /-
    case intro.intro
    α : Type u_1
    inst✝³ : CancelCommMonoidWithZero α
    inst✝² : UniqueFactorizationMonoid α
    inst✝¹ : DecidableEq (Associates α)
    inst✝ : (p : Associates α) → Decidable (Irreducible p)
    k : Nat
    a✝ : Nontrivial α
    a0 : α
    hz : Ne a0 0
    ha : Ne (Associates.mk a0) 0
    hk : ∀ (p : Associates α), Irreducible p → Dvd.dvd k (p.count (Associates.mk a …
    ⊢ Exists fun b => Eq (Associates.mk a0) (HPow.hPow b k)
  -/
  rw [factors_mk a0 hz] at hk
  have hk' : ∀ p, p ∈ factors' a0 → k ∣ (factors' a0).count p := by
    rintro p -
    have pp : p = ⟨p.val, p.2⟩ := by simp only [Subtype.coe_eta]
    rw [pp, ← count_some p.2]
    exact hk p.val p.2
  /-
    case intro.intro
    α : Type u_1
    inst✝³ : CancelCommMonoidWithZero α
    inst✝² : UniqueFactorizationMonoid α
    inst✝¹ : DecidableEq (Associates α)
    inst✝ : (p : Associates α) → Decidable (Irreducible p)
    k : Nat
    a✝ : Nontrivial α
    a0 : α
    hz : Ne a0 0
    ha : Ne (Associates.mk a0) 0
    hk : ∀ (p : Associates α), Irreducible p → Dvd.dvd k (p.count ↑(Associates.fac …
    hk' : ∀ (p : Subtype fun a => Irreducible a), Membership.mem (Associates.facto …
    ⊢ Exists fun b => Eq (Associates.mk a0) (HPow.hPow b k)
  -/
  obtain ⟨u, hu⟩ := Multiset.exists_smul_of_dvd_count _ hk'
  /-
    case intro.intro.intro
    α : Type u_1
    inst✝³ : CancelCommMonoidWithZero α
    inst✝² : UniqueFactorizationMonoid α
    inst✝¹ : DecidableEq (Associates α)
    inst✝ : (p : Associates α) → Decidable (Irreducible p)
    k : Nat
    a✝ : Nontrivial α
    a0 : α
    hz : Ne a0 0
    ha : Ne (Associates.mk a0) 0
    hk : ∀ (p : Associates α), Irreducible p → Dvd.dvd k (p.count ↑(Associates.fac …
    hk' : ∀ (p : Subtype fun a => Irreducible a), Membership.mem (Associates.facto …
    u : Multiset (Subtype fun a => Irreducible a)
    hu : Eq (Associates.factors' a0) (HSMul.hSMul k u)
    ⊢ Exists fun b => Eq (Associates.mk a0) (HPow.hPow b k)
  -/
  use FactorSet.prod (u : FactorSet α)
  /-
    case h
    α : Type u_1
    inst✝³ : CancelCommMonoidWithZero α
    inst✝² : UniqueFactorizationMonoid α
    inst✝¹ : DecidableEq (Associates α)
    inst✝ : (p : Associates α) → Decidable (Irreducible p)
    k : Nat
    a✝ : Nontrivial α
    a0 : α
    hz : Ne a0 0
    ha : Ne (Associates.mk a0) 0
    hk : ∀ (p : Associates α), Irreducible p → Dvd.dvd k (p.count ↑(Associates.fac …
    hk' : ∀ (p : Subtype fun a => Irreducible a), Membership.mem (Associates.facto …
    u : Multiset (Subtype fun a => Irreducible a)
    hu : Eq (Associates.factors' a0) (HSMul.hSMul k u)
    ⊢ Eq (Associates.mk a0) (HPow.hPow (Associates.FactorSet.prod ↑u) k)
  -/
  apply eq_of_factors_eq_factors
  /-
    case h.h
    α : Type u_1
    inst✝³ : CancelCommMonoidWithZero α
    inst✝² : UniqueFactorizationMonoid α
    inst✝¹ : DecidableEq (Associates α)
    inst✝ : (p : Associates α) → Decidable (Irreducible p)
    k : Nat
    a✝ : Nontrivial α
    a0 : α
    hz : Ne a0 0
    ha : Ne (Associates.mk a0) 0
    hk : ∀ (p : Associates α), Irreducible p → Dvd.dvd k (p.count ↑(Associates.fac …
    hk' : ∀ (p : Subtype fun a => Irreducible a), Membership.mem (Associates.facto …
    u : Multiset (Subtype fun a => Irreducible a)
    hu : Eq (Associates.factors' a0) (HSMul.hSMul k u)
    ⊢ Eq (Associates.mk a0).factors (HPow.hPow (Associates.FactorSet.prod ↑u) k).f …
  -/
  rw [pow_factors, prod_factors, factors_mk a0 hz, hu]
  /-
    case h.h
    α : Type u_1
    inst✝³ : CancelCommMonoidWithZero α
    inst✝² : UniqueFactorizationMonoid α
    inst✝¹ : DecidableEq (Associates α)
    inst✝ : (p : Associates α) → Decidable (Irreducible p)
    k : Nat
    a✝ : Nontrivial α
    a0 : α
    hz : Ne a0 0
    ha : Ne (Associates.mk a0) 0
    hk : ∀ (p : Associates α), Irreducible p → Dvd.dvd k (p.count ↑(Associates.fac …
    hk' : ∀ (p : Subtype fun a => Irreducible a), Membership.mem (Associates.facto …
    u : Multiset (Subtype fun a => Irreducible a)
    hu : Eq (Associates.factors' a0) (HSMul.hSMul k u)
    ⊢ Eq (↑(HSMul.hSMul k u)) (HSMul.hSMul k ↑u)
  -/
  exact WithBot.coe_nsmul u k
  /-
    🎉 no goals
  -/


/-- The only divisors of prime powers are prime powers. See `eq_pow_find_of_dvd_irreducible_pow`
for an explicit expression as a p-power (without using `count`). -/
theorem eq_pow_count_factors_of_dvd_pow {p a : Associates α}
    (hp : Irreducible p) {n : ℕ} (h : a ∣ p ^ n) : a = p ^ p.count a.factors := by
  /-
    α : Type u_1
    inst✝³ : CancelCommMonoidWithZero α
    inst✝² : UniqueFactorizationMonoid α
    inst✝¹ : DecidableEq (Associates α)
    inst✝ : (p : Associates α) → Decidable (Irreducible p)
    p a : Associates α
    hp : Irreducible p
    n : Nat
    h : Dvd.dvd a (HPow.hPow p n)
    ⊢ Eq a (HPow.hPow p (p.count a.factors))
  -/
  nontriviality α
  /-
    α : Type u_1
    inst✝³ : CancelCommMonoidWithZero α
    inst✝² : UniqueFactorizationMonoid α
    inst✝¹ : DecidableEq (Associates α)
    inst✝ : (p : Associates α) → Decidable (Irreducible p)
    p a : Associates α
    hp : Irreducible p
    n : Nat
    h : Dvd.dvd a (HPow.hPow p n)
    a✝ : Nontrivial α
    ⊢ Eq a (HPow.hPow p (p.count a.factors))
  -/
  have hph := pow_ne_zero n hp.ne_zero
  /-
    α : Type u_1
    inst✝³ : CancelCommMonoidWithZero α
    inst✝² : UniqueFactorizationMonoid α
    inst✝¹ : DecidableEq (Associates α)
    inst✝ : (p : Associates α) → Decidable (Irreducible p)
    p a : Associates α
    hp : Irreducible p
    n : Nat
    h : Dvd.dvd a (HPow.hPow p n)
    a✝ : Nontrivial α
    hph : Ne (HPow.hPow p n) 0
    ⊢ Eq a (HPow.hPow p (p.count a.factors))
  -/
  have ha := ne_zero_of_dvd_ne_zero hph h
  /-
    α : Type u_1
    inst✝³ : CancelCommMonoidWithZero α
    inst✝² : UniqueFactorizationMonoid α
    inst✝¹ : DecidableEq (Associates α)
    inst✝ : (p : Associates α) → Decidable (Irreducible p)
    p a : Associates α
    hp : Irreducible p
    n : Nat
    h : Dvd.dvd a (HPow.hPow p n)
    a✝ : Nontrivial α
    hph : Ne (HPow.hPow p n) 0
    ha : Ne a 0
    ⊢ Eq a (HPow.hPow p (p.count a.factors))
  -/
  apply eq_of_eq_counts ha (pow_ne_zero _ hp.ne_zero)
  have eq_zero_of_ne : ∀ q : Associates α, Irreducible q → q ≠ p → _ = 0 := fun q hq h' =>
    Nat.eq_zero_of_le_zero <| by
      convert count_le_count_of_le hph hq h
      symm
      rw [count_pow hp.ne_zero hq, count_eq_zero_of_ne hq hp h', mul_zero]
  /-
    α : Type u_1
    inst✝³ : CancelCommMonoidWithZero α
    inst✝² : UniqueFactorizationMonoid α
    inst✝¹ : DecidableEq (Associates α)
    inst✝ : (p : Associates α) → Decidable (Irreducible p)
    p a : Associates α
    hp : Irreducible p
    n : Nat
    h : Dvd.dvd a (HPow.hPow p n)
    a✝ : Nontrivial α
    hph : Ne (HPow.hPow p n) 0
    ha : Ne a 0
    eq_zero_of_ne : ∀ (q : Associates α), Irreducible q → Ne q p → Eq (q.count a.f …
    ⊢ ∀ (p_1 : Associates α), Irreducible p_1 → Eq (p_1.count a.factors) (p_1.coun …
  -/
  intro q hq
  /-
    α : Type u_1
    inst✝³ : CancelCommMonoidWithZero α
    inst✝² : UniqueFactorizationMonoid α
    inst✝¹ : DecidableEq (Associates α)
    inst✝ : (p : Associates α) → Decidable (Irreducible p)
    p a : Associates α
    hp : Irreducible p
    n : Nat
    h : Dvd.dvd a (HPow.hPow p n)
    a✝ : Nontrivial α
    hph : Ne (HPow.hPow p n) 0
    ha : Ne a 0
    eq_zero_of_ne : ∀ (q : Associates α), Irreducible q → Ne q p → Eq (q.count a.f …
    q : Associates α
    hq : Irreducible q
    ⊢ Eq (q.count a.factors) (q.count (HPow.hPow p (p.count a.factors)).factors)
  -/
  rw [count_pow hp.ne_zero hq]
  /-
    α : Type u_1
    inst✝³ : CancelCommMonoidWithZero α
    inst✝² : UniqueFactorizationMonoid α
    inst✝¹ : DecidableEq (Associates α)
    inst✝ : (p : Associates α) → Decidable (Irreducible p)
    p a : Associates α
    hp : Irreducible p
    n : Nat
    h : Dvd.dvd a (HPow.hPow p n)
    a✝ : Nontrivial α
    hph : Ne (HPow.hPow p n) 0
    ha : Ne a 0
    eq_zero_of_ne : ∀ (q : Associates α), Irreducible q → Ne q p → Eq (q.count a.f …
    q : Associates α
    hq : Irreducible q
    ⊢ Eq (q.count a.factors) (HMul.hMul (p.count a.factors) (q.count p.factors))
  -/
  by_cases h : q = p
    /-
      case pos
      α : Type u_1
      inst✝³ : CancelCommMonoidWithZero α
      inst✝² : UniqueFactorizationMonoid α
      inst✝¹ : DecidableEq (Associates α)
      inst✝ : (p : Associates α) → Decidable (Irreducible p)
      p a : Associates α
      hp : Irreducible p
      n : Nat
      h✝ : Dvd.dvd a (HPow.hPow p n)
      a✝ : Nontrivial α
      hph : Ne (HPow.hPow p n) 0
      ha : Ne a 0
      eq_zero_of_ne : ∀ (q : Associates α), Irreducible q → Ne q p → Eq (q.count a.f …
      q : Associates α
      hq : Irreducible q
      h : Eq q p
      ⊢ Eq (q.count a.factors) (HMul.hMul (p.count a.factors) (q.count p.factors))
    -/
  · rw [h, count_self hp, mul_one]
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      inst✝³ : CancelCommMonoidWithZero α
      inst✝² : UniqueFactorizationMonoid α
      inst✝¹ : DecidableEq (Associates α)
      inst✝ : (p : Associates α) → Decidable (Irreducible p)
      p a : Associates α
      hp : Irreducible p
      n : Nat
      h✝ : Dvd.dvd a (HPow.hPow p n)
      a✝ : Nontrivial α
      hph : Ne (HPow.hPow p n) 0
      ha : Ne a 0
      eq_zero_of_ne : ∀ (q : Associates α), Irreducible q → Ne q p → Eq (q.count a.f …
      q : Associates α
      hq : Irreducible q
      h : Not (Eq q p)
      ⊢ Eq (q.count a.factors) (HMul.hMul (p.count a.factors) (q.count p.factors))
    -/
  · rw [count_eq_zero_of_ne hq hp h, mul_zero, eq_zero_of_ne q hq h]
    /-
      🎉 no goals
    -/


theorem count_factors_eq_find_of_dvd_pow {a p : Associates α}
    (hp : Irreducible p) [∀ n : ℕ, Decidable (a ∣ p ^ n)] {n : ℕ} (h : a ∣ p ^ n) :
    @Nat.find (fun n => a ∣ p ^ n) _ ⟨n, h⟩ = p.count a.factors := by
  /-
    α : Type u_1
    inst✝⁴ : CancelCommMonoidWithZero α
    inst✝³ : UniqueFactorizationMonoid α
    inst✝² : DecidableEq (Associates α)
    inst✝¹ : (p : Associates α) → Decidable (Irreducible p)
    a p : Associates α
    hp : Irreducible p
    inst✝ : (n : Nat) → Decidable (Dvd.dvd a (HPow.hPow p n))
    n : Nat
    h : Dvd.dvd a (HPow.hPow p n)
    ⊢ Eq (Nat.find ⋯) (p.count a.factors)
  -/
  apply le_antisymm
    /-
      case a
      α : Type u_1
      inst✝⁴ : CancelCommMonoidWithZero α
      inst✝³ : UniqueFactorizationMonoid α
      inst✝² : DecidableEq (Associates α)
      inst✝¹ : (p : Associates α) → Decidable (Irreducible p)
      a p : Associates α
      hp : Irreducible p
      inst✝ : (n : Nat) → Decidable (Dvd.dvd a (HPow.hPow p n))
      n : Nat
      h : Dvd.dvd a (HPow.hPow p n)
      ⊢ LE.le (Nat.find ⋯) (p.count a.factors)
    -/
  · refine Nat.find_le ⟨1, ?_⟩
    /-
      case a
      α : Type u_1
      inst✝⁴ : CancelCommMonoidWithZero α
      inst✝³ : UniqueFactorizationMonoid α
      inst✝² : DecidableEq (Associates α)
      inst✝¹ : (p : Associates α) → Decidable (Irreducible p)
      a p : Associates α
      hp : Irreducible p
      inst✝ : (n : Nat) → Decidable (Dvd.dvd a (HPow.hPow p n))
      n : Nat
      h : Dvd.dvd a (HPow.hPow p n)
      ⊢ Eq (HPow.hPow p (p.count a.factors)) (HMul.hMul a 1)
    -/
    rw [mul_one]
    /-
      case a
      α : Type u_1
      inst✝⁴ : CancelCommMonoidWithZero α
      inst✝³ : UniqueFactorizationMonoid α
      inst✝² : DecidableEq (Associates α)
      inst✝¹ : (p : Associates α) → Decidable (Irreducible p)
      a p : Associates α
      hp : Irreducible p
      inst✝ : (n : Nat) → Decidable (Dvd.dvd a (HPow.hPow p n))
      n : Nat
      h : Dvd.dvd a (HPow.hPow p n)
      ⊢ Eq (HPow.hPow p (p.count a.factors)) a
    -/
    symm
    /-
      case a
      α : Type u_1
      inst✝⁴ : CancelCommMonoidWithZero α
      inst✝³ : UniqueFactorizationMonoid α
      inst✝² : DecidableEq (Associates α)
      inst✝¹ : (p : Associates α) → Decidable (Irreducible p)
      a p : Associates α
      hp : Irreducible p
      inst✝ : (n : Nat) → Decidable (Dvd.dvd a (HPow.hPow p n))
      n : Nat
      h : Dvd.dvd a (HPow.hPow p n)
      ⊢ Eq a (HPow.hPow p (p.count a.factors))
    -/
    exact eq_pow_count_factors_of_dvd_pow hp h
    /-
      🎉 no goals
    -/
    /-
      case a
      α : Type u_1
      inst✝⁴ : CancelCommMonoidWithZero α
      inst✝³ : UniqueFactorizationMonoid α
      inst✝² : DecidableEq (Associates α)
      inst✝¹ : (p : Associates α) → Decidable (Irreducible p)
      a p : Associates α
      hp : Irreducible p
      inst✝ : (n : Nat) → Decidable (Dvd.dvd a (HPow.hPow p n))
      n : Nat
      h : Dvd.dvd a (HPow.hPow p n)
      ⊢ LE.le (p.count a.factors) (Nat.find ⋯)
    -/
  · have hph := pow_ne_zero (@Nat.find (fun n => a ∣ p ^ n) _ ⟨n, h⟩) hp.ne_zero
    /-
      case a
      α : Type u_1
      inst✝⁴ : CancelCommMonoidWithZero α
      inst✝³ : UniqueFactorizationMonoid α
      inst✝² : DecidableEq (Associates α)
      inst✝¹ : (p : Associates α) → Decidable (Irreducible p)
      a p : Associates α
      hp : Irreducible p
      inst✝ : (n : Nat) → Decidable (Dvd.dvd a (HPow.hPow p n))
      n : Nat
      h : Dvd.dvd a (HPow.hPow p n)
      hph : Ne (HPow.hPow p (Nat.find ⋯)) 0
      ⊢ LE.le (p.count a.factors) (Nat.find ⋯)
    -/
    cases' subsingleton_or_nontrivial α with hα hα
      /-
        case a.inl
        α : Type u_1
        inst✝⁴ : CancelCommMonoidWithZero α
        inst✝³ : UniqueFactorizationMonoid α
        inst✝² : DecidableEq (Associates α)
        inst✝¹ : (p : Associates α) → Decidable (Irreducible p)
        a p : Associates α
        hp : Irreducible p
        inst✝ : (n : Nat) → Decidable (Dvd.dvd a (HPow.hPow p n))
        n : Nat
        h : Dvd.dvd a (HPow.hPow p n)
        hph : Ne (HPow.hPow p (Nat.find ⋯)) 0
        hα : Subsingleton α
        ⊢ LE.le (p.count a.factors) (Nat.find ⋯)
      -/
    · simp [eq_iff_true_of_subsingleton] at hph
      /-
        🎉 no goals
      -/
    /-
      case a.inr
      α : Type u_1
      inst✝⁴ : CancelCommMonoidWithZero α
      inst✝³ : UniqueFactorizationMonoid α
      inst✝² : DecidableEq (Associates α)
      inst✝¹ : (p : Associates α) → Decidable (Irreducible p)
      a p : Associates α
      hp : Irreducible p
      inst✝ : (n : Nat) → Decidable (Dvd.dvd a (HPow.hPow p n))
      n : Nat
      h : Dvd.dvd a (HPow.hPow p n)
      hph : Ne (HPow.hPow p (Nat.find ⋯)) 0
      hα : Nontrivial α
      ⊢ LE.le (p.count a.factors) (Nat.find ⋯)
    -/
    convert count_le_count_of_le hph hp (@Nat.find_spec (fun n => a ∣ p ^ n) _ ⟨n, h⟩)
    /-
      case h.e'_4
      α : Type u_1
      inst✝⁴ : CancelCommMonoidWithZero α
      inst✝³ : UniqueFactorizationMonoid α
      inst✝² : DecidableEq (Associates α)
      inst✝¹ : (p : Associates α) → Decidable (Irreducible p)
      a p : Associates α
      hp : Irreducible p
      inst✝ : (n : Nat) → Decidable (Dvd.dvd a (HPow.hPow p n))
      n : Nat
      h : Dvd.dvd a (HPow.hPow p n)
      hph : Ne (HPow.hPow p (Nat.find ⋯)) 0
      hα : Nontrivial α
      ⊢ Eq (Nat.find ⋯) (p.count (HPow.hPow p (Nat.find ⋯)).factors)
    -/
    rw [count_pow hp.ne_zero hp, count_self hp, mul_one]
    /-
      🎉 no goals
    -/


theorem eq_pow_of_mul_eq_pow {a b c : Associates α} (ha : a ≠ 0) (hb : b ≠ 0)
    (hab : ∀ d, d ∣ a → d ∣ b → ¬Prime d) {k : ℕ} (h : a * b = c ^ k) :
    ∃ d : Associates α, a = d ^ k := by
  classical
  nontriviality α
  by_cases hk0 : k = 0
  · use 1
    rw [hk0, pow_zero] at h ⊢
    apply (mul_eq_one.1 h).1
  · refine is_pow_of_dvd_count ha fun p hp ↦ ?_
    apply dvd_count_of_dvd_count_mul hb hp hab
    rw [h]
    apply dvd_count_pow _ hp
    rintro rfl
    rw [zero_pow hk0] at h
    cases mul_eq_zero.mp h <;> contradiction


/-- The only divisors of prime powers are prime powers. -/
theorem eq_pow_find_of_dvd_irreducible_pow {a p : Associates α} (hp : Irreducible p)
    [∀ n : ℕ, Decidable (a ∣ p ^ n)] {n : ℕ} (h : a ∣ p ^ n) :
    a = p ^ @Nat.find (fun n => a ∣ p ^ n) _ ⟨n, h⟩ := by
  /-
    α : Type u_1
    inst✝² : CancelCommMonoidWithZero α
    inst✝¹ : UniqueFactorizationMonoid α
    a p : Associates α
    hp : Irreducible p
    inst✝ : (n : Nat) → Decidable (Dvd.dvd a (HPow.hPow p n))
    n : Nat
    h : Dvd.dvd a (HPow.hPow p n)
    ⊢ Eq a (HPow.hPow p (Nat.find ⋯))
  -/
  classical rw [count_factors_eq_find_of_dvd_pow hp, ← eq_pow_count_factors_of_dvd_pow hp h]
  /-
    α : Type u_1
    inst✝² : CancelCommMonoidWithZero α
    inst✝¹ : UniqueFactorizationMonoid α
    a p : Associates α
    hp : Irreducible p
    inst✝ : (n : Nat) → Decidable (Dvd.dvd a (HPow.hPow p n))
    n : Nat
    h : Dvd.dvd a (HPow.hPow p n)
    ⊢ Dvd.dvd a (HPow.hPow p ?m.173200)
  -/
  exact h
  /-
    🎉 no goals
  -/


