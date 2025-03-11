/-- The lexicographic relation on `Π i : ι, β i`, where `ι` is ordered by `r`,
  and each `β i` is ordered by `s`. -/
protected def Lex (x y : ∀ i, β i) : Prop :=
  ∃ i, (∀ j, r j i → x j = y j) ∧ s (x i) (y i)

/- This unfortunately results in a type that isn't delta-reduced, so we keep the notation out of the
basic API, just in case -/

/-- The notation `Πₗ i, α i` refers to a pi type equipped with the lexicographic order. -/
notation3 (prettyPrint := false) "Πₗ "(...)", "r:(scoped p => Lex (∀ i, p i)) => r


@[simp]
theorem toLex_apply (x : ∀ i, β i) (i : ι) : toLex x i = x i :=
  rfl


@[simp]
theorem ofLex_apply (x : Lex (∀ i, β i)) (i : ι) : ofLex x i = x i :=
  rfl


theorem lex_lt_of_lt_of_preorder [∀ i, Preorder (β i)] {r} (hwf : WellFounded r) {x y : ∀ i, β i}
    (hlt : x < y) : ∃ i, (∀ j, r j i → x j ≤ y j ∧ y j ≤ x j) ∧ x i < y i :=
  let h' := Pi.lt_def.1 hlt
  let ⟨i, hi, hl⟩ := hwf.has_min _ h'.2
  ⟨i, fun j hj => ⟨h'.1 j, not_not.1 fun h => hl j (lt_of_le_not_le (h'.1 j) h) hj⟩, hi⟩


theorem lex_lt_of_lt [∀ i, PartialOrder (β i)] {r} (hwf : WellFounded r) {x y : ∀ i, β i}
    (hlt : x < y) : Pi.Lex r (@fun _ => (· < ·)) x y := by
  /-
    ι : Type u_1
    β : ι → Type u_2
    inst✝ : (i : ι) → PartialOrder (β i)
    r : ι → ι → Prop
    hwf : WellFounded r
    x y : (i : ι) → β i
    hlt : LT.lt x y
    ⊢ Pi.Lex r (fun x x1 x2 => LT.lt x1 x2) x y
  -/
  simp_rw [Pi.Lex, le_antisymm_iff]
  /-
    ι : Type u_1
    β : ι → Type u_2
    inst✝ : (i : ι) → PartialOrder (β i)
    r : ι → ι → Prop
    hwf : WellFounded r
    x y : (i : ι) → β i
    hlt : LT.lt x y
    ⊢ Exists fun i => And (∀ (j : ι), r j i → And (LE.le (x j) (y j)) (LE.le (y j) …
  -/
  exact lex_lt_of_lt_of_preorder hwf hlt
  /-
    🎉 no goals
  -/


theorem isTrichotomous_lex [∀ i, IsTrichotomous (β i) s] (wf : WellFounded r) :
    IsTrichotomous (∀ i, β i) (Pi.Lex r @s) :=
  { trichotomous := fun a b => by
      /-
        ι : Type u_1
        β : ι → Type u_2
        r : ι → ι → Prop
        s : {i : ι} → β i → β i → Prop
        inst✝ : ∀ (i : ι), IsTrichotomous (β i) s
        wf : WellFounded r
        a b : (i : ι) → β i
        ⊢ Or (Pi.Lex r s a b) (Or (Eq a b) (Pi.Lex r s b a))
      -/
      rcases eq_or_ne a b with hab | hab
        /-
          case inl
          ι : Type u_1
          β : ι → Type u_2
          r : ι → ι → Prop
          s : {i : ι} → β i → β i → Prop
          inst✝ : ∀ (i : ι), IsTrichotomous (β i) s
          wf : WellFounded r
          a b : (i : ι) → β i
          hab : Eq a b
          ⊢ Or (Pi.Lex r s a b) (Or (Eq a b) (Pi.Lex r s b a))
        -/
      · exact Or.inr (Or.inl hab)
        /-
          🎉 no goals
        -/
        /-
          case inr
          ι : Type u_1
          β : ι → Type u_2
          r : ι → ι → Prop
          s : {i : ι} → β i → β i → Prop
          inst✝ : ∀ (i : ι), IsTrichotomous (β i) s
          wf : WellFounded r
          a b : (i : ι) → β i
          hab : Ne a b
          ⊢ Or (Pi.Lex r s a b) (Or (Eq a b) (Pi.Lex r s b a))
        -/
      · rw [Function.ne_iff] at hab
        /-
          case inr
          ι : Type u_1
          β : ι → Type u_2
          r : ι → ι → Prop
          s : {i : ι} → β i → β i → Prop
          inst✝ : ∀ (i : ι), IsTrichotomous (β i) s
          wf : WellFounded r
          a b : (i : ι) → β i
          hab : Exists fun a_1 => Ne (a a_1) (b a_1)
          ⊢ Or (Pi.Lex r s a b) (Or (Eq a b) (Pi.Lex r s b a))
        -/
        let i := wf.min _ hab
        have hri : ∀ j, r j i → a j = b j := by
          intro j
          rw [← not_imp_not]
          exact fun h' => wf.not_lt_min _ _ h'
        /-
          case inr
          ι : Type u_1
          β : ι → Type u_2
          r : ι → ι → Prop
          s : {i : ι} → β i → β i → Prop
          inst✝ : ∀ (i : ι), IsTrichotomous (β i) s
          wf : WellFounded r
          a b : (i : ι) → β i
          hab : Exists fun a_1 => Ne (a a_1) (b a_1)
          i : ι := wf.min (fun x => Eq (a x) (b x) → False) hab
          hri : ∀ (j : ι), r j i → Eq (a j) (b j)
          ⊢ Or (Pi.Lex r s a b) (Or (Eq a b) (Pi.Lex r s b a))
        -/
        have hne : a i ≠ b i := wf.min_mem _ hab
        /-
          case inr
          ι : Type u_1
          β : ι → Type u_2
          r : ι → ι → Prop
          s : {i : ι} → β i → β i → Prop
          inst✝ : ∀ (i : ι), IsTrichotomous (β i) s
          wf : WellFounded r
          a b : (i : ι) → β i
          hab : Exists fun a_1 => Ne (a a_1) (b a_1)
          i : ι := wf.min (fun x => Eq (a x) (b x) → False) hab
          hri : ∀ (j : ι), r j i → Eq (a j) (b j)
          hne : Ne (a i) (b i)
          ⊢ Or (Pi.Lex r s a b) (Or (Eq a b) (Pi.Lex r s b a))
        -/
        cases' trichotomous_of s (a i) (b i) with hi hi
        exacts [Or.inl ⟨i, hri, hi⟩,
          Or.inr <| Or.inr <| ⟨i, fun j hj => (hri j hj).symm, hi.resolve_left hne⟩] }


instance [LT ι] [∀ a, LT (β a)] : LT (Lex (∀ i, β i)) :=
  ⟨Pi.Lex (· < ·) @fun _ => (· < ·)⟩


instance Lex.isStrictOrder [LinearOrder ι] [∀ a, PartialOrder (β a)] :
    IsStrictOrder (Lex (∀ i, β i)) (· < ·) where
  irrefl := fun a ⟨k, _, hk₂⟩ => lt_irrefl (a k) hk₂
  trans := by
    /-
      ι : Type u_1
      β : ι → Type u_2
      r : ι → ι → Prop
      s : {i : ι} → β i → β i → Prop
      inst✝¹ : LinearOrder ι
      inst✝ : (a : ι) → PartialOrder (β a)
      ⊢ ∀ (a b c : Lex ((i : ι) → β i)), LT.lt a b → LT.lt b c → LT.lt a c
    -/
    rintro a b c ⟨N₁, lt_N₁, a_lt_b⟩ ⟨N₂, lt_N₂, b_lt_c⟩
    /-
      case intro.intro.intro.intro
      ι : Type u_1
      β : ι → Type u_2
      r : ι → ι → Prop
      s : {i : ι} → β i → β i → Prop
      inst✝¹ : LinearOrder ι
      inst✝ : (a : ι) → PartialOrder (β a)
      a b c : Lex ((i : ι) → β i)
      N₁ : ι
      lt_N₁ : ∀ (j : ι), (fun x1 x2 => LT.lt x1 x2) j N₁ → Eq (a j) (b j)
      a_lt_b : LT.lt (a N₁) (b N₁)
      N₂ : ι
      lt_N₂ : ∀ (j : ι), (fun x1 x2 => LT.lt x1 x2) j N₂ → Eq (b j) (c j)
      b_lt_c : LT.lt (b N₂) (c N₂)
      ⊢ LT.lt a c
    -/
    rcases lt_trichotomy N₁ N₂ with (H | rfl | H)
    exacts [⟨N₁, fun j hj => (lt_N₁ _ hj).trans (lt_N₂ _ <| hj.trans H), lt_N₂ _ H ▸ a_lt_b⟩,
      ⟨N₁, fun j hj => (lt_N₁ _ hj).trans (lt_N₂ _ hj), a_lt_b.trans b_lt_c⟩,
      ⟨N₂, fun j hj => (lt_N₁ _ (hj.trans H)).trans (lt_N₂ _ hj), (lt_N₁ _ H).symm ▸ b_lt_c⟩]


instance [LinearOrder ι] [∀ a, PartialOrder (β a)] : PartialOrder (Lex (∀ i, β i)) :=
  partialOrderOfSO (· < ·)


/-- `Πₗ i, α i` is a linear order if the original order is well-founded. -/
noncomputable instance [LinearOrder ι] [WellFoundedLT ι] [∀ a, LinearOrder (β a)] :
    LinearOrder (Lex (∀ i, β i)) :=
  @linearOrderOfSTO (Πₗ i, β i) (· < ·)
    { trichotomous := (isTrichotomous_lex _ _ IsWellFounded.wf).1 } (Classical.decRel _)


theorem toLex_monotone : Monotone (@toLex (∀ i, β i)) := fun a b h =>
  or_iff_not_imp_left.2 fun hne =>
    let ⟨i, hi, hl⟩ := IsWellFounded.wf.has_min (r := (· < ·)) { i | a i ≠ b i }
      (Function.ne_iff.1 hne)
    ⟨i, fun j hj => by
      /-
        ι : Type u_1
        β : ι → Type u_2
        inst✝² : LinearOrder ι
        inst✝¹ : WellFoundedLT ι
        inst✝ : (i : ι) → PartialOrder (β i)
        a b : (i : ι) → β i
        h : LE.le a b
        hne : Not (Eq (toLex a) (toLex b))
        i : ι
        hi : Membership.mem (setOf fun i => Ne (a i) (b i)) i
        hl : ∀ (x : ι), Membership.mem (setOf fun i => Ne (a i) (b i)) x → Not (LT.lt  …
        j : ι
        hj : (fun x1 x2 => LT.lt x1 x2) j i
        ⊢ Eq (toLex a j) (toLex b j)
      -/
      contrapose! hl
      /-
        ι : Type u_1
        β : ι → Type u_2
        inst✝² : LinearOrder ι
        inst✝¹ : WellFoundedLT ι
        inst✝ : (i : ι) → PartialOrder (β i)
        a b : (i : ι) → β i
        h : LE.le a b
        hne : Not (Eq (toLex a) (toLex b))
        i : ι
        hi : Membership.mem (setOf fun i => Ne (a i) (b i)) i
        j : ι
        hj : (fun x1 x2 => LT.lt x1 x2) j i
        hl : Ne (toLex a j) (toLex b j)
        ⊢ Exists fun x => And (Membership.mem (setOf fun i => Ne (a i) (b i)) x) (LT.l …
      -/
      exact ⟨j, hl, hj⟩, (h i).lt_of_ne hi⟩
      /-
        🎉 no goals
      -/


theorem toLex_strictMono : StrictMono (@toLex (∀ i, β i)) := fun a b h =>
  let ⟨i, hi, hl⟩ := IsWellFounded.wf.has_min (r := (· < ·)) { i | a i ≠ b i }
    (Function.ne_iff.1 h.ne)
  ⟨i, fun j hj => by
    /-
      ι : Type u_1
      β : ι → Type u_2
      inst✝² : LinearOrder ι
      inst✝¹ : WellFoundedLT ι
      inst✝ : (i : ι) → PartialOrder (β i)
      a b : (i : ι) → β i
      h : LT.lt a b
      i : ι
      hi : Membership.mem (setOf fun i => Ne (a i) (b i)) i
      hl : ∀ (x : ι), Membership.mem (setOf fun i => Ne (a i) (b i)) x → Not (LT.lt  …
      j : ι
      hj : (fun x1 x2 => LT.lt x1 x2) j i
      ⊢ Eq (toLex a j) (toLex b j)
    -/
    contrapose! hl
    /-
      ι : Type u_1
      β : ι → Type u_2
      inst✝² : LinearOrder ι
      inst✝¹ : WellFoundedLT ι
      inst✝ : (i : ι) → PartialOrder (β i)
      a b : (i : ι) → β i
      h : LT.lt a b
      i : ι
      hi : Membership.mem (setOf fun i => Ne (a i) (b i)) i
      j : ι
      hj : (fun x1 x2 => LT.lt x1 x2) j i
      hl : Ne (toLex a j) (toLex b j)
      ⊢ Exists fun x => And (Membership.mem (setOf fun i => Ne (a i) (b i)) x) (LT.l …
    -/
    exact ⟨j, hl, hj⟩, (h.le i).lt_of_ne hi⟩
    /-
      🎉 no goals
    -/


@[simp]
theorem lt_toLex_update_self_iff : toLex x < toLex (update x i a) ↔ x i < a := by
  /-
    ι : Type u_1
    β : ι → Type u_2
    inst✝² : LinearOrder ι
    inst✝¹ : WellFoundedLT ι
    inst✝ : (i : ι) → PartialOrder (β i)
    x : (i : ι) → β i
    i : ι
    a : β i
    ⊢ Iff (LT.lt (toLex x) (toLex (Function.update x i a))) (LT.lt (x i) a)
  -/
  refine ⟨?_, fun h => toLex_strictMono <| lt_update_self_iff.2 h⟩
  /-
    ι : Type u_1
    β : ι → Type u_2
    inst✝² : LinearOrder ι
    inst✝¹ : WellFoundedLT ι
    inst✝ : (i : ι) → PartialOrder (β i)
    x : (i : ι) → β i
    i : ι
    a : β i
    ⊢ LT.lt (toLex x) (toLex (Function.update x i a)) → LT.lt (x i) a
  -/
  rintro ⟨j, hj, h⟩
  /-
    case intro.intro
    ι : Type u_1
    β : ι → Type u_2
    inst✝² : LinearOrder ι
    inst✝¹ : WellFoundedLT ι
    inst✝ : (i : ι) → PartialOrder (β i)
    x : (i : ι) → β i
    i : ι
    a : β i
    j : ι
    hj : ∀ (j_1 : ι), (fun x1 x2 => LT.lt x1 x2) j_1 j → Eq (toLex x j_1) (toLex ( …
    h : LT.lt (toLex x j) (toLex (Function.update x i a) j)
    ⊢ LT.lt (x i) a
  -/
  dsimp at h
  obtain rfl : j = i := by
    by_contra H
    rw [update_of_ne H] at h
    exact h.false
  /-
    case intro.intro
    ι : Type u_1
    β : ι → Type u_2
    inst✝² : LinearOrder ι
    inst✝¹ : WellFoundedLT ι
    inst✝ : (i : ι) → PartialOrder (β i)
    x : (i : ι) → β i
    j : ι
    a : β j
    hj : ∀ (j_1 : ι), (fun x1 x2 => LT.lt x1 x2) j_1 j → Eq (toLex x j_1) (toLex ( …
    h : LT.lt (x j) (Function.update x j a j)
    ⊢ LT.lt (x j) a
  -/
  rwa [update_self] at h
  /-
    🎉 no goals
  -/


@[simp]
theorem toLex_update_lt_self_iff : toLex (update x i a) < toLex x ↔ a < x i := by
  /-
    ι : Type u_1
    β : ι → Type u_2
    inst✝² : LinearOrder ι
    inst✝¹ : WellFoundedLT ι
    inst✝ : (i : ι) → PartialOrder (β i)
    x : (i : ι) → β i
    i : ι
    a : β i
    ⊢ Iff (LT.lt (toLex (Function.update x i a)) (toLex x)) (LT.lt a (x i))
  -/
  refine ⟨?_, fun h => toLex_strictMono <| update_lt_self_iff.2 h⟩
  /-
    ι : Type u_1
    β : ι → Type u_2
    inst✝² : LinearOrder ι
    inst✝¹ : WellFoundedLT ι
    inst✝ : (i : ι) → PartialOrder (β i)
    x : (i : ι) → β i
    i : ι
    a : β i
    ⊢ LT.lt (toLex (Function.update x i a)) (toLex x) → LT.lt a (x i)
  -/
  rintro ⟨j, hj, h⟩
  /-
    case intro.intro
    ι : Type u_1
    β : ι → Type u_2
    inst✝² : LinearOrder ι
    inst✝¹ : WellFoundedLT ι
    inst✝ : (i : ι) → PartialOrder (β i)
    x : (i : ι) → β i
    i : ι
    a : β i
    j : ι
    hj : ∀ (j_1 : ι), (fun x1 x2 => LT.lt x1 x2) j_1 j → Eq (toLex (Function.updat …
    h : LT.lt (toLex (Function.update x i a) j) (toLex x j)
    ⊢ LT.lt a (x i)
  -/
  dsimp at h
  obtain rfl : j = i := by
    by_contra H
    rw [update_of_ne H] at h
    exact h.false
  /-
    case intro.intro
    ι : Type u_1
    β : ι → Type u_2
    inst✝² : LinearOrder ι
    inst✝¹ : WellFoundedLT ι
    inst✝ : (i : ι) → PartialOrder (β i)
    x : (i : ι) → β i
    j : ι
    a : β j
    hj : ∀ (j_1 : ι), (fun x1 x2 => LT.lt x1 x2) j_1 j → Eq (toLex (Function.updat …
    h : LT.lt (Function.update x j a j) (x j)
    ⊢ LT.lt a (x j)
  -/
  rwa [update_self] at h
  /-
    🎉 no goals
  -/


@[simp]
theorem le_toLex_update_self_iff : toLex x ≤ toLex (update x i a) ↔ x i ≤ a := by
  /-
    ι : Type u_1
    β : ι → Type u_2
    inst✝² : LinearOrder ι
    inst✝¹ : WellFoundedLT ι
    inst✝ : (i : ι) → PartialOrder (β i)
    x : (i : ι) → β i
    i : ι
    a : β i
    ⊢ Iff (LE.le (toLex x) (toLex (Function.update x i a))) (LE.le (x i) a)
  -/
  simp_rw [le_iff_lt_or_eq, lt_toLex_update_self_iff, toLex_inj, eq_update_self_iff]
  /-
    🎉 no goals
  -/


@[simp]
theorem toLex_update_le_self_iff : toLex (update x i a) ≤ toLex x ↔ a ≤ x i := by
  /-
    ι : Type u_1
    β : ι → Type u_2
    inst✝² : LinearOrder ι
    inst✝¹ : WellFoundedLT ι
    inst✝ : (i : ι) → PartialOrder (β i)
    x : (i : ι) → β i
    i : ι
    a : β i
    ⊢ Iff (LE.le (toLex (Function.update x i a)) (toLex x)) (LE.le a (x i))
  -/
  simp_rw [le_iff_lt_or_eq, toLex_update_lt_self_iff, toLex_inj, update_eq_self_iff]
  /-
    🎉 no goals
  -/


instance [LinearOrder ι] [WellFoundedLT ι] [∀ a, PartialOrder (β a)] [∀ a, OrderBot (β a)] :
    OrderBot (Lex (∀ a, β a)) where
  bot := toLex ⊥
  bot_le _ := toLex_monotone bot_le


instance [LinearOrder ι] [WellFoundedLT ι] [∀ a, PartialOrder (β a)] [∀ a, OrderTop (β a)] :
    OrderTop (Lex (∀ a, β a)) where
  top := toLex ⊤
  le_top _ := toLex_monotone le_top


instance [LinearOrder ι] [WellFoundedLT ι] [∀ a, PartialOrder (β a)]
    [∀ a, BoundedOrder (β a)] : BoundedOrder (Lex (∀ a, β a)) :=
  { }


instance [Preorder ι] [∀ i, LT (β i)] [∀ i, DenselyOrdered (β i)] :
    DenselyOrdered (Lex (∀ i, β i)) :=
  ⟨by
    /-
      ι : Type u_1
      β : ι → Type u_2
      r : ι → ι → Prop
      s : {i : ι} → β i → β i → Prop
      inst✝² : Preorder ι
      inst✝¹ : (i : ι) → LT (β i)
      inst✝ : ∀ (i : ι), DenselyOrdered (β i)
      ⊢ ∀ (a₁ a₂ : Lex ((i : ι) → β i)), LT.lt a₁ a₂ → Exists fun a => And (LT.lt a₁ …
    -/
    rintro _ a₂ ⟨i, h, hi⟩
    /-
      case intro.intro
      ι : Type u_1
      β : ι → Type u_2
      r : ι → ι → Prop
      s : {i : ι} → β i → β i → Prop
      inst✝² : Preorder ι
      inst✝¹ : (i : ι) → LT (β i)
      inst✝ : ∀ (i : ι), DenselyOrdered (β i)
      a₁✝ a₂ : Lex ((i : ι) → β i)
      i : ι
      h : ∀ (j : ι), (fun x1 x2 => LT.lt x1 x2) j i → Eq (a₁✝ j) (a₂ j)
      hi : LT.lt (a₁✝ i) (a₂ i)
      ⊢ Exists fun a => And (LT.lt a₁✝ a) (LT.lt a a₂)
    -/
    obtain ⟨a, ha₁, ha₂⟩ := exists_between hi
    classical
      refine ⟨Function.update a₂ _ a, ⟨i, fun j hj => ?_, ?_⟩, i, fun j hj => ?_, ?_⟩
      · rw [h j hj]
        dsimp only at hj
        rw [Function.update_of_ne hj.ne a]
      · rwa [Function.update_self i a]
      · rw [Function.update_of_ne hj.ne a]
      · rwa [Function.update_self i a]⟩


theorem Lex.noMaxOrder' [Preorder ι] [∀ i, LT (β i)] (i : ι) [NoMaxOrder (β i)] :
    NoMaxOrder (Lex (∀ i, β i)) :=
  ⟨fun a => by
    /-
      ι : Type u_1
      β : ι → Type u_2
      inst✝² : Preorder ι
      inst✝¹ : (i : ι) → LT (β i)
      i : ι
      inst✝ : NoMaxOrder (β i)
      a : Lex ((i : ι) → β i)
      ⊢ Exists fun b => LT.lt a b
    -/
    let ⟨b, hb⟩ := exists_gt (a i)
    classical
    exact ⟨Function.update a i b, i, fun j hj =>
      (Function.update_of_ne hj.ne b a).symm, by rwa [Function.update_self i b]⟩⟩


instance [LinearOrder ι] [WellFoundedLT ι] [Nonempty ι] [∀ i, PartialOrder (β i)]
    [∀ i, NoMaxOrder (β i)] : NoMaxOrder (Lex (∀ i, β i)) :=
  ⟨fun a =>
    let ⟨_, hb⟩ := exists_gt (ofLex a)
    ⟨_, toLex_strictMono hb⟩⟩


instance [LinearOrder ι] [WellFoundedLT ι] [Nonempty ι] [∀ i, PartialOrder (β i)]
    [∀ i, NoMinOrder (β i)] : NoMinOrder (Lex (∀ i, β i)) :=
  ⟨fun a =>
    let ⟨_, hb⟩ := exists_lt (ofLex a)
    ⟨_, toLex_strictMono hb⟩⟩


/-- If we swap two strictly decreasing values in a function, then the result is lexicographically
smaller than the original function. -/
theorem lex_desc {α} [Preorder ι] [DecidableEq ι] [Preorder α] {f : ι → α} {i j : ι} (h₁ : i ≤ j)
    (h₂ : f j < f i) : toLex (f ∘ Equiv.swap i j) < toLex f :=
  ⟨i, fun _ hik => congr_arg f (Equiv.swap_apply_of_ne_of_ne hik.ne (hik.trans_le h₁).ne), by
    /-
      ι : Type u_1
      α : Type u_3
      inst✝² : Preorder ι
      inst✝¹ : DecidableEq ι
      inst✝ : Preorder α
      f : ι → α
      i j : ι
      h₁ : LE.le i j
      h₂ : LT.lt (f j) (f i)
      ⊢ (fun x x1 x2 => LT.lt x1 x2) i (toLex (Function.comp f ⇑(Equiv.swap i j)) i) …
    -/
    simpa only [Pi.toLex_apply, Function.comp_apply, Equiv.swap_apply_left] using h₂⟩
    /-
      🎉 no goals
    -/


