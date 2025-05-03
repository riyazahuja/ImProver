/-- Auxiliary definition for the `CoeSort` instance. This prevents the `CoeOut m α`
instance from inadvertently applying to other sigma types. -/
def Multiset.ToType (m : Multiset α) : Type _ := (x : α) × Fin (m.count x)


/-- Create a type that has the same number of elements as the multiset.
Terms of this type are triples `⟨x, ⟨i, h⟩⟩` where `x : α`, `i : ℕ`, and `h : i < m.count x`.
This way repeated elements of a multiset appear multiple times from different values of `i`. -/
instance : CoeSort (Multiset α) (Type _) := ⟨Multiset.ToType⟩


/-- Constructor for terms of the coercion of `m` to a type.
This helps Lean pick up the correct instances. -/
@[reducible, match_pattern]
def Multiset.mkToType (m : Multiset α) (x : α) (i : Fin (m.count x)) : m :=
  ⟨x, i⟩


/-- As a convenience, there is a coercion from `m : Type*` to `α` by projecting onto the first
component. -/
instance instCoeSortMultisetType.instCoeOutToType : CoeOut m α :=
  ⟨fun x ↦ x.1⟩

-- Porting note: syntactic equality

-- Syntactic equality

-- @[simp] -- Porting note (https://github.com/leanprover-community/mathlib4/issues/10685): dsimp can prove this

theorem Multiset.coe_mk {x : α} {i : Fin (m.count x)} : ↑(m.mkToType x i) = x :=
  rfl


                                                                             /-
                                                                               α : Type u_1
                                                                               inst✝ : DecidableEq α
                                                                               m : Multiset α
                                                                               x : m.ToType
                                                                               ⊢ LT.lt 0 (Multiset.count x.fst m)
                                                                             -/
@[simp] lemma Multiset.coe_mem {x : m} : ↑x ∈ m := Multiset.count_pos.mp (by have := x.2.2; omega)
                                                                                            /-
                                                                                              🎉 no goals
                                                                                            -/


@[simp]
protected theorem Multiset.forall_coe (p : m → Prop) :
    (∀ x : m, p x) ↔ ∀ (x : α) (i : Fin (m.count x)), p ⟨x, i⟩ :=
  Sigma.forall


@[simp]
protected theorem Multiset.exists_coe (p : m → Prop) :
    (∃ x : m, p x) ↔ ∃ (x : α) (i : Fin (m.count x)), p ⟨x, i⟩ :=
  Sigma.exists


instance : Fintype { p : α × ℕ | p.2 < m.count p.1 } :=
  Fintype.ofFinset
    (m.toFinset.biUnion fun x ↦ (Finset.range (m.count x)).map ⟨Prod.mk x, Prod.mk.inj_left x⟩)
    (by
      /-
        α : Type u_1
        inst✝ : DecidableEq α
        m : Multiset α
        ⊢ ∀ (x : Prod α Nat), Iff (Membership.mem (m.toFinset.biUnion fun x => Finset. …
      -/
      rintro ⟨x, i⟩
      simp only [Finset.mem_biUnion, Multiset.mem_toFinset, Finset.mem_map, Finset.mem_range,
        Function.Embedding.coeFn_mk, Prod.mk.inj_iff, Set.mem_setOf_eq]
      /-
        case mk
        α : Type u_1
        inst✝ : DecidableEq α
        m : Multiset α
        x : α
        i : Nat
        ⊢ Iff (Exists fun a => And (Membership.mem m a) (Exists fun a_1 => And (LT.lt  …
      -/
      simp only [← and_assoc, exists_eq_right, and_iff_right_iff_imp]
      /-
        case mk
        α : Type u_1
        inst✝ : DecidableEq α
        m : Multiset α
        x : α
        i : Nat
        ⊢ LT.lt i (Multiset.count x m) → Membership.mem m x
      -/
      exact fun h ↦ Multiset.count_pos.mp (by omega))
      /-
        🎉 no goals
      -/


/-- Construct a finset whose elements enumerate the elements of the multiset `m`.
The `ℕ` component is used to differentiate between equal elements: if `x` appears `n` times
then `(x, 0)`, ..., and `(x, n-1)` appear in the `Finset`. -/
def Multiset.toEnumFinset (m : Multiset α) : Finset (α × ℕ) :=
  { p : α × ℕ | p.2 < m.count p.1 }.toFinset


@[simp]
theorem Multiset.mem_toEnumFinset (m : Multiset α) (p : α × ℕ) :
    p ∈ m.toEnumFinset ↔ p.2 < m.count p.1 :=
  Set.mem_toFinset


theorem Multiset.mem_of_mem_toEnumFinset {p : α × ℕ} (h : p ∈ m.toEnumFinset) : p.1 ∈ m :=
                                                                 /-
                                                                   α : Type u_1
                                                                   inst✝ : DecidableEq α
                                                                   m : Multiset α
                                                                   p : Prod α Nat
                                                                   h : Membership.mem m.toEnumFinset p
                                                                   this : LT.lt p.2 (Multiset.count p.1 m)
                                                                   ⊢ LT.lt 0 (Multiset.count p.1 m)
                                                                 -/
  have := (m.mem_toEnumFinset p).mp h; Multiset.count_pos.mp (by omega)
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


@[simp] lemma toEnumFinset_filter_eq (m : Multiset α) (a : α) :
                                                                            /-
                                                                              α : Type u_1
                                                                              inst✝ : DecidableEq α
                                                                              m : Multiset α
                                                                              a : α
                                                                              ⊢ Eq (Finset.filter (fun x => Eq x.1 a) m.toEnumFinset) (SProd.sprod (Singleto …
                                                                            -/
    m.toEnumFinset.filter (·.1 = a) = {a} ×ˢ Finset.range (m.count a) := by aesop
                                                                            /-
                                                                              🎉 no goals
                                                                            -/


@[simp] lemma map_toEnumFinset_fst (m : Multiset α) : m.toEnumFinset.val.map Prod.fst = m := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    m : Multiset α
    ⊢ Eq (Multiset.map Prod.fst m.toEnumFinset.val) m
  -/
  ext a; simp [count_map, ← Finset.filter_val, eq_comm (a := a)]
         /-
           🎉 no goals
         -/


@[simp] lemma image_toEnumFinset_fst (m : Multiset α) :
    m.toEnumFinset.image Prod.fst = m.toFinset := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    m : Multiset α
    ⊢ Eq (Finset.image Prod.fst m.toEnumFinset) m.toFinset
  -/
  rw [Finset.image, Multiset.map_toEnumFinset_fst]
  /-
    🎉 no goals
  -/


@[simp] lemma map_fst_le_of_subset_toEnumFinset {s : Finset (α × ℕ)} (hsm : s ⊆ m.toEnumFinset) :
    s.1.map Prod.fst ≤ m := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    m : Multiset α
    s : Finset (Prod α Nat)
    hsm : HasSubset.Subset s m.toEnumFinset
    ⊢ LE.le (Multiset.map Prod.fst s.val) m
  -/
  simp_rw [le_iff_count, count_map]
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    m : Multiset α
    s : Finset (Prod α Nat)
    hsm : HasSubset.Subset s m.toEnumFinset
    ⊢ ∀ (a : α), LE.le (Multiset.filter (fun a_1 => Eq a a_1.1) s.val).card (Multi …
  -/
  rintro a
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    m : Multiset α
    s : Finset (Prod α Nat)
    hsm : HasSubset.Subset s m.toEnumFinset
    a : α
    ⊢ LE.le (Multiset.filter (fun a_1 => Eq a a_1.1) s.val).card (Multiset.count a …
  -/
  obtain ha | ha := (s.1.filter fun x ↦ a = x.1).card.eq_zero_or_pos
    /-
      case inl
      α : Type u_1
      inst✝ : DecidableEq α
      m : Multiset α
      s : Finset (Prod α Nat)
      hsm : HasSubset.Subset s m.toEnumFinset
      a : α
      ha : Eq (Multiset.filter (fun x => Eq a x.1) s.val).card 0
      ⊢ LE.le (Multiset.filter (fun a_1 => Eq a a_1.1) s.val).card (Multiset.count a …
    -/
  · rw [ha]
    /-
      case inl
      α : Type u_1
      inst✝ : DecidableEq α
      m : Multiset α
      s : Finset (Prod α Nat)
      hsm : HasSubset.Subset s m.toEnumFinset
      a : α
      ha : Eq (Multiset.filter (fun x => Eq a x.1) s.val).card 0
      ⊢ LE.le 0 (Multiset.count a m)
    -/
    exact Nat.zero_le _
    /-
      🎉 no goals
    -/
  obtain ⟨n, han, hn⟩ : ∃ n ≥ card (s.1.filter fun x ↦ a = x.1) - 1, (a, n) ∈ s := by
    by_contra! h
    replace h : s.filter (·.1 = a) ⊆ {a} ×ˢ .range (card (s.1.filter fun x ↦ a = x.1) - 1) := by
      simpa (config := { contextual := true }) [forall_swap (β := _ = a), Finset.subset_iff,
        imp_not_comm, not_le, Nat.lt_sub_iff_add_lt] using h
    have : card (s.1.filter fun x ↦ a = x.1) ≤ card (s.1.filter fun x ↦ a = x.1) - 1 := by
      simpa [Finset.card, eq_comm] using Finset.card_mono h
    omega
  /-
    case inr.intro.intro
    α : Type u_1
    inst✝ : DecidableEq α
    m : Multiset α
    s : Finset (Prod α Nat)
    hsm : HasSubset.Subset s m.toEnumFinset
    a : α
    ha : GT.gt (Multiset.filter (fun x => Eq a x.1) s.val).card 0
    n : Nat
    han : GE.ge n (HSub.hSub (Multiset.filter (fun x => Eq a x.1) s.val).card 1)
    hn : Membership.mem s { fst := a, snd := n }
    ⊢ LE.le (Multiset.filter (fun a_1 => Eq a a_1.1) s.val).card (Multiset.count a …
  -/
  exact Nat.le_of_pred_lt (han.trans_lt <| by simpa using hsm hn)
  /-
    🎉 no goals
  -/


@[mono]
theorem Multiset.toEnumFinset_mono {m₁ m₂ : Multiset α} (h : m₁ ≤ m₂) :
    m₁.toEnumFinset ⊆ m₂.toEnumFinset := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    m₁ m₂ : Multiset α
    h : LE.le m₁ m₂
    ⊢ HasSubset.Subset m₁.toEnumFinset m₂.toEnumFinset
  -/
  intro p
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    m₁ m₂ : Multiset α
    h : LE.le m₁ m₂
    p : Prod α Nat
    ⊢ Membership.mem m₁.toEnumFinset p → Membership.mem m₂.toEnumFinset p
  -/
  simp only [Multiset.mem_toEnumFinset]
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    m₁ m₂ : Multiset α
    h : LE.le m₁ m₂
    p : Prod α Nat
    ⊢ LT.lt p.2 (Multiset.count p.1 m₁) → LT.lt p.2 (Multiset.count p.1 m₂)
  -/
  exact gt_of_ge_of_gt (Multiset.le_iff_count.mp h p.1)
  /-
    🎉 no goals
  -/


@[simp]
theorem Multiset.toEnumFinset_subset_iff {m₁ m₂ : Multiset α} :
    m₁.toEnumFinset ⊆ m₂.toEnumFinset ↔ m₁ ≤ m₂ :=
              /-
                α : Type u_1
                inst✝ : DecidableEq α
                m₁ m₂ : Multiset α
                h : HasSubset.Subset m₁.toEnumFinset m₂.toEnumFinset
                ⊢ LE.le m₁ m₂
              -/
  ⟨fun h ↦ by simpa using map_fst_le_of_subset_toEnumFinset h, Multiset.toEnumFinset_mono⟩
              /-
                🎉 no goals
              -/


/-- The embedding from a multiset into `α × ℕ` where the second coordinate enumerates repeats.
If you are looking for the function `m → α`, that would be plain `(↑)`. -/
@[simps]
def Multiset.coeEmbedding (m : Multiset α) : m ↪ α × ℕ where
  toFun x := (x, x.2)
  inj' := by
    /-
      α : Type u_1
      inst✝ : DecidableEq α
      m✝ m : Multiset α
      ⊢ Function.Injective fun x => { fst := x.fst, snd := ↑x.snd }
    -/
    intro ⟨x, i, hi⟩ ⟨y, j, hj⟩
    /-
      α : Type u_1
      inst✝ : DecidableEq α
      m✝ m : Multiset α
      x : α
      i : Nat
      hi : LT.lt i (Multiset.count x m)
      y : α
      j : Nat
      hj : LT.lt j (Multiset.count y m)
      ⊢ Eq ((fun x => { fst := x.fst, snd := ↑x.snd }) ⟨x, ⟨i, hi⟩⟩) ((fun x => { fs …
    -/
    rintro ⟨⟩
    /-
      case refl
      α : Type u_1
      inst✝ : DecidableEq α
      m✝ m : Multiset α
      x : α
      i : Nat
      hi hj : LT.lt i (Multiset.count x m)
      ⊢ Eq ⟨x, ⟨i, hi⟩⟩ ⟨x, ⟨i, hj⟩⟩
    -/
    rfl
    /-
      🎉 no goals
    -/


/-- Another way to coerce a `Multiset` to a type is to go through `m.toEnumFinset` and coerce
that `Finset` to a type. -/
@[simps]
def Multiset.coeEquiv (m : Multiset α) : m ≃ m.toEnumFinset where
  toFun x :=
    ⟨m.coeEmbedding x, by
      /-
        α : Type u_1
        inst✝ : DecidableEq α
        m✝ m : Multiset α
        x : m.ToType
        ⊢ Membership.mem m.toEnumFinset (m.coeEmbedding x)
      -/
      rw [Multiset.mem_toEnumFinset]
      /-
        α : Type u_1
        inst✝ : DecidableEq α
        m✝ m : Multiset α
        x : m.ToType
        ⊢ LT.lt (m.coeEmbedding x).2 (Multiset.count (m.coeEmbedding x).1 m)
      -/
      exact x.2.2⟩
      /-
        🎉 no goals
      -/
  invFun x :=
    ⟨x.1.1, x.1.2, by
      /-
        α : Type u_1
        inst✝ : DecidableEq α
        m✝ m : Multiset α
        x : Subtype fun x => Membership.mem m.toEnumFinset x
        ⊢ LT.lt (↑x).2 (Multiset.count (↑x).1 m)
      -/
      rw [← Multiset.mem_toEnumFinset]
      /-
        α : Type u_1
        inst✝ : DecidableEq α
        m✝ m : Multiset α
        x : Subtype fun x => Membership.mem m.toEnumFinset x
        ⊢ Membership.mem m.toEnumFinset ↑x
      -/
      exact x.2⟩
      /-
        🎉 no goals
      -/
  left_inv := by
    /-
      α : Type u_1
      inst✝ : DecidableEq α
      m✝ m : Multiset α
      ⊢ Function.LeftInverse (fun x => ⟨(↑x).1, ⟨(↑x).2, ⋯⟩⟩) fun x => ⟨m.coeEmbeddi …
    -/
    rintro ⟨x, i, h⟩
    /-
      case mk.mk
      α : Type u_1
      inst✝ : DecidableEq α
      m✝ m : Multiset α
      x : α
      i : Nat
      h : LT.lt i (Multiset.count x m)
      ⊢ Eq ((fun x => ⟨(↑x).1, ⟨(↑x).2, ⋯⟩⟩) ((fun x => ⟨m.coeEmbedding x, ⋯⟩) ⟨x, ⟨ …
    -/
    rfl
    /-
      🎉 no goals
    -/
  right_inv := by
    /-
      α : Type u_1
      inst✝ : DecidableEq α
      m✝ m : Multiset α
      ⊢ Function.RightInverse (fun x => ⟨(↑x).1, ⟨(↑x).2, ⋯⟩⟩) fun x => ⟨m.coeEmbedd …
    -/
    rintro ⟨⟨x, i⟩, h⟩
    /-
      case mk.mk
      α : Type u_1
      inst✝ : DecidableEq α
      m✝ m : Multiset α
      x : α
      i : Nat
      h : Membership.mem m.toEnumFinset { fst := x, snd := i }
      ⊢ Eq ((fun x => ⟨m.coeEmbedding x, ⋯⟩) ((fun x => ⟨(↑x).1, ⟨(↑x).2, ⋯⟩⟩) ⟨{ fs …
    -/
    rfl
    /-
      🎉 no goals
    -/


@[simp]
theorem Multiset.toEmbedding_coeEquiv_trans (m : Multiset α) :
                                                                                       /-
                                                                                         α : Type u_1
                                                                                         inst✝ : DecidableEq α
                                                                                         m : Multiset α
                                                                                         ⊢ Eq (m.coeEquiv.toEmbedding.trans (Function.Embedding.subtype fun x => Member …
                                                                                       -/
                                                                                               /-
                                                                                                 🎉 no goals
                                                                                               -/
    m.coeEquiv.toEmbedding.trans (Function.Embedding.subtype _) = m.coeEmbedding := by ext <;> rfl
                                                                                               /-
                                                                                                 🎉 no goals
                                                                                               -/


@[irreducible]
instance Multiset.fintypeCoe : Fintype m :=
  Fintype.ofEquiv m.toEnumFinset m.coeEquiv.symm


theorem Multiset.map_univ_coeEmbedding (m : Multiset α) :
    (Finset.univ : Finset m).map m.coeEmbedding = m.toEnumFinset := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    m : Multiset α
    ⊢ Eq (Finset.map m.coeEmbedding Finset.univ) m.toEnumFinset
  -/
  ext ⟨x, i⟩
  simp only [Fin.exists_iff, Finset.mem_map, Finset.mem_univ, Multiset.coeEmbedding_apply,
    Prod.mk.inj_iff, exists_true_left, Multiset.exists_coe, Multiset.coe_mk, Fin.val_mk,
    exists_prop, exists_eq_right_right, exists_eq_right, Multiset.mem_toEnumFinset, true_and]


@[simp]
theorem Multiset.map_univ_coe (m : Multiset α) :
    (Finset.univ : Finset m).val.map (fun x : m ↦ (x : α)) = m := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    m : Multiset α
    ⊢ Eq (Multiset.map (fun x => x.fst) Finset.univ.val) m
  -/
  have := m.map_toEnumFinset_fst
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    m : Multiset α
    this : Eq (Multiset.map Prod.fst m.toEnumFinset.val) m
    ⊢ Eq (Multiset.map (fun x => x.fst) Finset.univ.val) m
  -/
  rw [← m.map_univ_coeEmbedding] at this
  simpa only [Finset.map_val, Multiset.coeEmbedding_apply, Multiset.map_map,
    Function.comp_apply] using this


@[simp]
theorem Multiset.map_univ {β : Type*} (m : Multiset α) (f : α → β) :
    ((Finset.univ : Finset m).val.map fun (x : m) ↦ f (x : α)) = m.map f := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    β : Type u_2
    m : Multiset α
    f : α → β
    ⊢ Eq (Multiset.map (fun x => f x.fst) Finset.univ.val) (Multiset.map f m)
  -/
  erw [← Multiset.map_map, Multiset.map_univ_coe]
  /-
    🎉 no goals
  -/


@[simp]
theorem Multiset.card_toEnumFinset (m : Multiset α) : m.toEnumFinset.card = Multiset.card m := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    m : Multiset α
    ⊢ Eq m.toEnumFinset.card m.card
  -/
  rw [Finset.card, ← Multiset.card_map Prod.fst m.toEnumFinset.val]
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    m : Multiset α
    ⊢ Eq (Multiset.map Prod.fst m.toEnumFinset.val).card m.card
  -/
  congr
  /-
    case e_s
    α : Type u_1
    inst✝ : DecidableEq α
    m : Multiset α
    ⊢ Eq (Multiset.map Prod.fst m.toEnumFinset.val) m
  -/
  exact m.map_toEnumFinset_fst
  /-
    🎉 no goals
  -/


@[simp]
theorem Multiset.card_coe (m : Multiset α) : Fintype.card m = Multiset.card m := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    m : Multiset α
    ⊢ Eq (Fintype.card m.ToType) m.card
  -/
  rw [Fintype.card_congr m.coeEquiv]
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    m : Multiset α
    ⊢ Eq (Fintype.card (Subtype fun x => Membership.mem m.toEnumFinset x)) m.card
  -/
  simp only [Fintype.card_coe, card_toEnumFinset]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem Multiset.prod_eq_prod_coe [CommMonoid α] (m : Multiset α) : m.prod = ∏ x : m, (x : α) := by
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : CommMonoid α
    m : Multiset α
    ⊢ Eq m.prod (Finset.univ.prod fun x => x.fst)
  -/
  congr
  /-
    case e_a
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : CommMonoid α
    m : Multiset α
    ⊢ Eq m (Multiset.map (fun x => x.fst) Finset.univ.val)
  -/
  simp
  /-
    🎉 no goals
  -/


@[to_additive]
theorem Multiset.prod_eq_prod_toEnumFinset [CommMonoid α] (m : Multiset α) :
    m.prod = ∏ x ∈ m.toEnumFinset, x.1 := by
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : CommMonoid α
    m : Multiset α
    ⊢ Eq m.prod (m.toEnumFinset.prod fun x => x.1)
  -/
  congr
  /-
    case e_a
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : CommMonoid α
    m : Multiset α
    ⊢ Eq m (Multiset.map (fun x => x.1) m.toEnumFinset.val)
  -/
  simp
  /-
    🎉 no goals
  -/


@[to_additive]
theorem Multiset.prod_toEnumFinset {β : Type*} [CommMonoid β] (m : Multiset α) (f : α → ℕ → β) :
    ∏ x ∈ m.toEnumFinset, f x.1 x.2 = ∏ x : m, f x x.2 := by
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    β : Type u_2
    inst✝ : CommMonoid β
    m : Multiset α
    f : α → Nat → β
    ⊢ Eq (m.toEnumFinset.prod fun x => f x.1 x.2) (Finset.univ.prod fun x => f x.f …
  -/
  rw [Fintype.prod_equiv m.coeEquiv (fun x ↦ f x x.2) fun x ↦ f x.1.1 x.1.2]
    /-
      α : Type u_1
      inst✝¹ : DecidableEq α
      β : Type u_2
      inst✝ : CommMonoid β
      m : Multiset α
      f : α → Nat → β
      ⊢ Eq (m.toEnumFinset.prod fun x => f x.1 x.2) (Finset.univ.prod fun x => f (↑x …
    -/
  · rw [← m.toEnumFinset.prod_coe_sort fun x ↦ f x.1 x.2]
    /-
      🎉 no goals
    -/
    /-
      α : Type u_1
      inst✝¹ : DecidableEq α
      β : Type u_2
      inst✝ : CommMonoid β
      m : Multiset α
      f : α → Nat → β
      ⊢ ∀ (x : m.ToType), Eq (f x.fst ↑x.snd) (f (↑(m.coeEquiv x)).1 (↑(m.coeEquiv x …
    -/
  · intro x
    /-
      α : Type u_1
      inst✝¹ : DecidableEq α
      β : Type u_2
      inst✝ : CommMonoid β
      m : Multiset α
      f : α → Nat → β
      x : m.ToType
      ⊢ Eq (f x.fst ↑x.snd) (f (↑(m.coeEquiv x)).1 (↑(m.coeEquiv x)).2)
    -/
    rfl
    /-
      🎉 no goals
    -/

