/-- Type synonym of `Finset α` equipped with the colexicographic order rather than the inclusion
order. -/
@[ext]
structure Colex (α) where
  /-- `toColex` is the "identity" function between `Finset α` and `Finset.Colex α`. -/
  toColex ::
  /-- `ofColex` is the "identity" function between `Finset.Colex α` and `Finset α`. -/
  (ofColex : Finset α)

-- TODO: Why can't we export?
--export Colex (toColex)


instance : Inhabited (Colex α) := ⟨⟨∅⟩⟩


@[simp] lemma toColex_ofColex (s : Colex α) : toColex (ofColex s) = s := rfl

lemma ofColex_toColex (s : Finset α) : ofColex (toColex s) = s := rfl

                                                                         /-
                                                                           α : Type u_1
                                                                           s t : Finset α
                                                                           ⊢ Iff (Eq { ofColex := s } { ofColex := t }) (Eq s t)
                                                                         -/
lemma toColex_inj {s t : Finset α} : toColex s = toColex t ↔ s = t := by simp
                                                                         /-
                                                                           🎉 no goals
                                                                         -/

@[simp]
                                                                        /-
                                                                          α : Type u_1
                                                                          s t : Finset.Colex α
                                                                          ⊢ Iff (Eq s.ofColex t.ofColex) (Eq s t)
                                                                        -/
lemma ofColex_inj {s t : Colex α} : ofColex s = ofColex t ↔ s = t := by cases s; cases t; simp
                                                                                          /-
                                                                                            🎉 no goals
                                                                                          -/

                                                                                /-
                                                                                  α : Type u_1
                                                                                  s t : Finset α
                                                                                  ⊢ Iff (Ne { ofColex := s } { ofColex := t }) (Ne s t)
                                                                                -/
lemma toColex_ne_toColex {s t : Finset α} : toColex s ≠ toColex t ↔ s ≠ t := by simp
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/

                                                                               /-
                                                                                 α : Type u_1
                                                                                 s t : Finset.Colex α
                                                                                 ⊢ Iff (Ne s.ofColex t.ofColex) (Ne s t)
                                                                               -/
lemma ofColex_ne_ofColex {s t : Colex α} : ofColex s ≠ ofColex t ↔ s ≠ t := by simp
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/


lemma toColex_injective : Injective (toColex : Finset α → Colex α) := fun _ _ ↦ toColex_inj.1

lemma ofColex_injective : Injective (ofColex : Colex α → Finset α) := fun _ _ ↦ ofColex_inj.1


instance instLE : LE (Colex α) where
  le s t := ∀ ⦃a⦄, a ∈ ofColex s → a ∉ ofColex t → ∃ b, b ∈ ofColex t ∧ b ∉ ofColex s ∧ a ≤ b

-- TODO: This lemma is weirdly useful given how strange its statement is.
-- Is there a nicer statement? Should this lemma be made public?

private lemma trans_aux (hst : toColex s ≤ toColex t) (htu : toColex t ≤ toColex u)
    (has : a ∈ s) (hat : a ∉ t) : ∃ b, b ∈ u ∧ b ∉ s ∧ a ≤ b := by
  classical
  let s' : Finset α := {b ∈ s | b ∉ t ∧ a ≤ b}
  have ⟨b, hb, hbmax⟩ := exists_maximal s' ⟨a, by simp [s', has, hat]⟩
  simp only [s', mem_filter, and_imp] at hb hbmax
  have ⟨c, hct, hcs, hbc⟩ := hst hb.1 hb.2.1
  by_cases hcu : c ∈ u
  · exact ⟨c, hcu, hcs, hb.2.2.trans hbc⟩
  have ⟨d, hdu, hdt, hcd⟩ := htu hct hcu
  have had : a ≤ d := hb.2.2.trans <| hbc.trans hcd
  refine ⟨d, hdu, fun hds ↦ ?_, had⟩
  exact hbmax d hds hdt had <| hbc.trans_lt <| hcd.lt_of_ne <| ne_of_mem_of_not_mem hct hdt


private lemma antisymm_aux (hst : toColex s ≤ toColex t) (hts : toColex t ≤ toColex s) : s ⊆ t := by
  /-
    α : Type u_1
    inst✝ : PartialOrder α
    s t : Finset α
    hst : LE.le { ofColex := s } { ofColex := t }
    hts : LE.le { ofColex := t } { ofColex := s }
    ⊢ HasSubset.Subset s t
  -/
  intro a has
  /-
    α : Type u_1
    inst✝ : PartialOrder α
    s t : Finset α
    hst : LE.le { ofColex := s } { ofColex := t }
    hts : LE.le { ofColex := t } { ofColex := s }
    a : α
    has : Membership.mem s a
    ⊢ Membership.mem t a
  -/
  by_contra! hat
  /-
    α : Type u_1
    inst✝ : PartialOrder α
    s t : Finset α
    hst : LE.le { ofColex := s } { ofColex := t }
    hts : LE.le { ofColex := t } { ofColex := s }
    a : α
    has : Membership.mem s a
    hat : Not (Membership.mem t a)
    ⊢ False
  -/
  have ⟨_b, hb₁, hb₂, _⟩ := trans_aux hst hts has hat
  /-
    α : Type u_1
    inst✝ : PartialOrder α
    s t : Finset α
    hst : LE.le { ofColex := s } { ofColex := t }
    hts : LE.le { ofColex := t } { ofColex := s }
    a : α
    has : Membership.mem s a
    hat : Not (Membership.mem t a)
    _b : α
    hb₁ : Membership.mem s _b
    hb₂ : Not (Membership.mem s _b)
    right✝ : LE.le a _b
    ⊢ False
  -/
  exact hb₂ hb₁
  /-
    🎉 no goals
  -/


instance instPartialOrder : PartialOrder (Colex α) where
  le_refl _ _ ha ha' := (ha' ha).elim
  le_antisymm _ _ hst hts := Colex.ext <| (antisymm_aux hst hts).antisymm (antisymm_aux hts hst)
  le_trans s t u hst htu a has hau := by
    /-
      α : Type u_1
      β : Type u_2
      inst✝¹ : PartialOrder α
      inst✝ : PartialOrder β
      f : α → β
      𝒜 𝒜₁ 𝒜₂ : Finset (Finset α)
      s✝ t✝ u✝ : Finset α
      a✝ b : α
      s t u : Finset.Colex α
      hst : LE.le s t
      htu : LE.le t u
      a : α
      has : Membership.mem s.ofColex a
      hau : Not (Membership.mem u.ofColex a)
      ⊢ Exists fun b => And (Membership.mem u.ofColex b) (And (Not (Membership.mem s …
    -/
    by_cases hat : a ∈ ofColex t
      /-
        case pos
        α : Type u_1
        β : Type u_2
        inst✝¹ : PartialOrder α
        inst✝ : PartialOrder β
        f : α → β
        𝒜 𝒜₁ 𝒜₂ : Finset (Finset α)
        s✝ t✝ u✝ : Finset α
        a✝ b : α
        s t u : Finset.Colex α
        hst : LE.le s t
        htu : LE.le t u
        a : α
        has : Membership.mem s.ofColex a
        hau : Not (Membership.mem u.ofColex a)
        hat : Membership.mem t.ofColex a
        ⊢ Exists fun b => And (Membership.mem u.ofColex b) (And (Not (Membership.mem s …
      -/
    · have ⟨b, hbu, hbt, hab⟩ := htu hat hau
      /-
        case pos
        α : Type u_1
        β : Type u_2
        inst✝¹ : PartialOrder α
        inst✝ : PartialOrder β
        f : α → β
        𝒜 𝒜₁ 𝒜₂ : Finset (Finset α)
        s✝ t✝ u✝ : Finset α
        a✝ b✝ : α
        s t u : Finset.Colex α
        hst : LE.le s t
        htu : LE.le t u
        a : α
        has : Membership.mem s.ofColex a
        hau : Not (Membership.mem u.ofColex a)
        hat : Membership.mem t.ofColex a
        b : α
        hbu : Membership.mem u.ofColex b
        hbt : Not (Membership.mem t.ofColex b)
        hab : LE.le a b
        ⊢ Exists fun b => And (Membership.mem u.ofColex b) (And (Not (Membership.mem s …
      -/
      by_cases hbs : b ∈ ofColex s
        /-
          case pos
          α : Type u_1
          β : Type u_2
          inst✝¹ : PartialOrder α
          inst✝ : PartialOrder β
          f : α → β
          𝒜 𝒜₁ 𝒜₂ : Finset (Finset α)
          s✝ t✝ u✝ : Finset α
          a✝ b✝ : α
          s t u : Finset.Colex α
          hst : LE.le s t
          htu : LE.le t u
          a : α
          has : Membership.mem s.ofColex a
          hau : Not (Membership.mem u.ofColex a)
          hat : Membership.mem t.ofColex a
          b : α
          hbu : Membership.mem u.ofColex b
          hbt : Not (Membership.mem t.ofColex b)
          hab : LE.le a b
          hbs : Membership.mem s.ofColex b
          ⊢ Exists fun b => And (Membership.mem u.ofColex b) (And (Not (Membership.mem s …
        -/
      · have ⟨c, hcu, hcs, hbc⟩ := trans_aux hst htu hbs hbt
        /-
          case pos
          α : Type u_1
          β : Type u_2
          inst✝¹ : PartialOrder α
          inst✝ : PartialOrder β
          f : α → β
          𝒜 𝒜₁ 𝒜₂ : Finset (Finset α)
          s✝ t✝ u✝ : Finset α
          a✝ b✝ : α
          s t u : Finset.Colex α
          hst : LE.le s t
          htu : LE.le t u
          a : α
          has : Membership.mem s.ofColex a
          hau : Not (Membership.mem u.ofColex a)
          hat : Membership.mem t.ofColex a
          b : α
          hbu : Membership.mem u.ofColex b
          hbt : Not (Membership.mem t.ofColex b)
          hab : LE.le a b
          hbs : Membership.mem s.ofColex b
          c : α
          hcu : Membership.mem u.ofColex c
          hcs : Not (Membership.mem s.ofColex c)
          hbc : LE.le b c
          ⊢ Exists fun b => And (Membership.mem u.ofColex b) (And (Not (Membership.mem s …
        -/
        exact ⟨c, hcu, hcs, hab.trans hbc⟩
        /-
          🎉 no goals
        -/
        /-
          case neg
          α : Type u_1
          β : Type u_2
          inst✝¹ : PartialOrder α
          inst✝ : PartialOrder β
          f : α → β
          𝒜 𝒜₁ 𝒜₂ : Finset (Finset α)
          s✝ t✝ u✝ : Finset α
          a✝ b✝ : α
          s t u : Finset.Colex α
          hst : LE.le s t
          htu : LE.le t u
          a : α
          has : Membership.mem s.ofColex a
          hau : Not (Membership.mem u.ofColex a)
          hat : Membership.mem t.ofColex a
          b : α
          hbu : Membership.mem u.ofColex b
          hbt : Not (Membership.mem t.ofColex b)
          hab : LE.le a b
          hbs : Not (Membership.mem s.ofColex b)
          ⊢ Exists fun b => And (Membership.mem u.ofColex b) (And (Not (Membership.mem s …
        -/
      · exact ⟨b, hbu, hbs, hab⟩
        /-
          🎉 no goals
        -/
      /-
        case neg
        α : Type u_1
        β : Type u_2
        inst✝¹ : PartialOrder α
        inst✝ : PartialOrder β
        f : α → β
        𝒜 𝒜₁ 𝒜₂ : Finset (Finset α)
        s✝ t✝ u✝ : Finset α
        a✝ b : α
        s t u : Finset.Colex α
        hst : LE.le s t
        htu : LE.le t u
        a : α
        has : Membership.mem s.ofColex a
        hau : Not (Membership.mem u.ofColex a)
        hat : Not (Membership.mem t.ofColex a)
        ⊢ Exists fun b => And (Membership.mem u.ofColex b) (And (Not (Membership.mem s …
      -/
    · exact trans_aux hst htu has hat
      /-
        🎉 no goals
      -/


lemma le_def {s t : Colex α} :
    s ≤ t ↔ ∀ ⦃a⦄, a ∈ ofColex s → a ∉ ofColex t → ∃ b, b ∈ ofColex t ∧ b ∉ ofColex s ∧ a ≤ b :=
  Iff.rfl


lemma toColex_le_toColex :
    toColex s ≤ toColex t ↔ ∀ ⦃a⦄, a ∈ s → a ∉ t → ∃ b, b ∈ t ∧ b ∉ s ∧ a ≤ b := Iff.rfl


lemma toColex_lt_toColex :
    toColex s < toColex t ↔ s ≠ t ∧ ∀ ⦃a⦄, a ∈ s → a ∉ t → ∃ b, b ∈ t ∧ b ∉ s ∧ a ≤ b := by
  /-
    α : Type u_1
    inst✝ : PartialOrder α
    s t : Finset α
    ⊢ Iff (LT.lt { ofColex := s } { ofColex := t }) (And (Ne s t) (∀ ⦃a : α⦄, Memb …
  -/
  simp [lt_iff_le_and_ne, toColex_le_toColex, and_comm]
  /-
    🎉 no goals
  -/


/-- If `s ⊆ t`, then `s ≤ t` in the colex order. Note the converse does not hold, as inclusion does
not form a linear order. -/
lemma toColex_mono : Monotone (toColex : Finset α → Colex α) :=
  fun _s _t hst _a has hat ↦ (hat <| hst has).elim


/-- If `s ⊂ t`, then `s < t` in the colex order. Note the converse does not hold, as inclusion does
not form a linear order. -/
lemma toColex_strictMono : StrictMono (toColex : Finset α → Colex α) :=
  toColex_mono.strictMono_of_injective toColex_injective


/-- If `s ⊆ t`, then `s ≤ t` in the colex order. Note the converse does not hold, as inclusion does
not form a linear order. -/
lemma toColex_le_toColex_of_subset (h : s ⊆ t) : toColex s ≤ toColex t := toColex_mono h


/-- If `s ⊂ t`, then `s < t` in the colex order. Note the converse does not hold, as inclusion does
not form a linear order. -/
lemma toColex_lt_toColex_of_ssubset (h : s ⊂ t) : toColex s < toColex t := toColex_strictMono h


instance instOrderBot : OrderBot (Colex α) where
  bot := toColex ∅
                      /-
                        α : Type u_1
                        β : Type u_2
                        inst✝¹ : PartialOrder α
                        inst✝ : PartialOrder β
                        f : α → β
                        𝒜 𝒜₁ 𝒜₂ : Finset (Finset α)
                        s✝ t u : Finset α
                        a✝ b : α
                        s : Finset.Colex α
                        a : α
                        ha : Membership.mem Bot.bot.ofColex a
                        ⊢ Not (Membership.mem s.ofColex a) → Exists fun b => And (Membership.mem s.ofC …
                      -/
  bot_le s a ha := by cases ha
                      /-
                        🎉 no goals
                      -/


@[simp] lemma toColex_empty : toColex (∅ : Finset α) = ⊥ := rfl

@[simp] lemma ofColex_bot : ofColex (⊥ : Colex α) = ∅ := rfl


/-- If `s ≤ t` in colex, and all elements in `t` are small, then all elements in `s` are small. -/
lemma forall_le_mono (hst : toColex s ≤ toColex t) (ht : ∀ b ∈ t, b ≤ a) : ∀ b ∈ s, b ≤ a := by
  /-
    α : Type u_1
    inst✝ : PartialOrder α
    s t : Finset α
    a : α
    hst : LE.le { ofColex := s } { ofColex := t }
    ht : ∀ (b : α), Membership.mem t b → LE.le b a
    ⊢ ∀ (b : α), Membership.mem s b → LE.le b a
  -/
  rintro b hb
  /-
    α : Type u_1
    inst✝ : PartialOrder α
    s t : Finset α
    a : α
    hst : LE.le { ofColex := s } { ofColex := t }
    ht : ∀ (b : α), Membership.mem t b → LE.le b a
    b : α
    hb : Membership.mem s b
    ⊢ LE.le b a
  -/
  by_cases b ∈ t
    /-
      case pos
      α : Type u_1
      inst✝ : PartialOrder α
      s t : Finset α
      a : α
      hst : LE.le { ofColex := s } { ofColex := t }
      ht : ∀ (b : α), Membership.mem t b → LE.le b a
      b : α
      hb : Membership.mem s b
      h✝ : Membership.mem t b
      ⊢ LE.le b a
    -/
  · exact ht _ ‹_›
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      inst✝ : PartialOrder α
      s t : Finset α
      a : α
      hst : LE.le { ofColex := s } { ofColex := t }
      ht : ∀ (b : α), Membership.mem t b → LE.le b a
      b : α
      hb : Membership.mem s b
      h✝ : Not (Membership.mem t b)
      ⊢ LE.le b a
    -/
  · obtain ⟨c, hct, -, hbc⟩ := hst hb ‹_›
    /-
      case neg.intro.intro.intro
      α : Type u_1
      inst✝ : PartialOrder α
      s t : Finset α
      a : α
      hst : LE.le { ofColex := s } { ofColex := t }
      ht : ∀ (b : α), Membership.mem t b → LE.le b a
      b : α
      hb : Membership.mem s b
      h✝ : Not (Membership.mem t b)
      c : α
      hct : Membership.mem { ofColex := t }.ofColex c
      hbc : LE.le b c
      ⊢ LE.le b a
    -/
    exact hbc.trans <| ht _ hct
    /-
      🎉 no goals
    -/


/-- If `s ≤ t` in colex, and all elements in `t` are small, then all elements in `s` are small. -/
lemma forall_lt_mono (hst : toColex s ≤ toColex t) (ht : ∀ b ∈ t, b < a) : ∀ b ∈ s, b < a := by
  /-
    α : Type u_1
    inst✝ : PartialOrder α
    s t : Finset α
    a : α
    hst : LE.le { ofColex := s } { ofColex := t }
    ht : ∀ (b : α), Membership.mem t b → LT.lt b a
    ⊢ ∀ (b : α), Membership.mem s b → LT.lt b a
  -/
  rintro b hb
  /-
    α : Type u_1
    inst✝ : PartialOrder α
    s t : Finset α
    a : α
    hst : LE.le { ofColex := s } { ofColex := t }
    ht : ∀ (b : α), Membership.mem t b → LT.lt b a
    b : α
    hb : Membership.mem s b
    ⊢ LT.lt b a
  -/
  by_cases b ∈ t
    /-
      case pos
      α : Type u_1
      inst✝ : PartialOrder α
      s t : Finset α
      a : α
      hst : LE.le { ofColex := s } { ofColex := t }
      ht : ∀ (b : α), Membership.mem t b → LT.lt b a
      b : α
      hb : Membership.mem s b
      h✝ : Membership.mem t b
      ⊢ LT.lt b a
    -/
  · exact ht _ ‹_›
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      inst✝ : PartialOrder α
      s t : Finset α
      a : α
      hst : LE.le { ofColex := s } { ofColex := t }
      ht : ∀ (b : α), Membership.mem t b → LT.lt b a
      b : α
      hb : Membership.mem s b
      h✝ : Not (Membership.mem t b)
      ⊢ LT.lt b a
    -/
  · obtain ⟨c, hct, -, hbc⟩ := hst hb ‹_›
    /-
      case neg.intro.intro.intro
      α : Type u_1
      inst✝ : PartialOrder α
      s t : Finset α
      a : α
      hst : LE.le { ofColex := s } { ofColex := t }
      ht : ∀ (b : α), Membership.mem t b → LT.lt b a
      b : α
      hb : Membership.mem s b
      h✝ : Not (Membership.mem t b)
      c : α
      hct : Membership.mem { ofColex := t }.ofColex c
      hbc : LE.le b c
      ⊢ LT.lt b a
    -/
    exact hbc.trans_lt <| ht _ hct
    /-
      🎉 no goals
    -/


/-- `s ≤ {a}` in colex iff all elements of `s` are strictly less than `a`, except possibly `a` in
which case `s = {a}`. -/
lemma toColex_le_singleton : toColex s ≤ toColex {a} ↔ ∀ b ∈ s, b ≤ a ∧ (a ∈ s → b = a) := by
  /-
    α : Type u_1
    inst✝ : PartialOrder α
    s : Finset α
    a : α
    ⊢ Iff (LE.le { ofColex := s } { ofColex := Singleton.singleton a }) (∀ (b : α) …
  -/
  simp only [toColex_le_toColex, mem_singleton, and_assoc, exists_eq_left]
  /-
    α : Type u_1
    inst✝ : PartialOrder α
    s : Finset α
    a : α
    ⊢ Iff (∀ ⦃a_1 : α⦄, Membership.mem s a_1 → Not (Eq a_1 a) → And (Not (Membersh …
  -/
                                                                          /-
                                                                            🎉 no goals
                                                                          -/
  refine forall₂_congr fun b _ ↦ ?_; obtain rfl | hba := eq_or_ne b a <;> aesop
                                                                          /-
                                                                            🎉 no goals
                                                                          -/


/-- `s < {a}` in colex iff all elements of `s` are strictly less than `a`. -/
lemma toColex_lt_singleton : toColex s < toColex {a} ↔ ∀ b ∈ s, b < a := by
  /-
    α : Type u_1
    inst✝ : PartialOrder α
    s : Finset α
    a : α
    ⊢ Iff (LT.lt { ofColex := s } { ofColex := Singleton.singleton a }) (∀ (b : α) …
  -/
  rw [lt_iff_le_and_ne, toColex_le_singleton, toColex_ne_toColex]
  refine ⟨fun h b hb ↦ (h.1 _ hb).1.lt_of_ne ?_,
                                                                                       /-
                                                                                         case refine_1
                                                                                         α : Type u_1
                                                                                         inst✝ : PartialOrder α
                                                                                         s : Finset α
                                                                                         a : α
                                                                                         h : And (∀ (b : α), Membership.mem s b → And (LE.le b a) (Membership.mem s a → …
                                                                                         b : α
                                                                                         hb : Membership.mem s b
                                                                                         ⊢ Ne b a
                                                                                       -/
    fun h ↦ ⟨fun b hb ↦ ⟨(h _ hb).le, fun ha ↦ (lt_irrefl _ <| h _ ha).elim⟩, ?_⟩⟩ <;> rintro rfl
    /-
      case refine_1
      α : Type u_1
      inst✝ : PartialOrder α
      s : Finset α
      b : α
      hb : Membership.mem s b
      h : And (∀ (b_1 : α), Membership.mem s b_1 → And (LE.le b_1 b) (Membership.mem …
      ⊢ False
    -/
  · refine h.2 <| eq_singleton_iff_unique_mem.2 ⟨hb, fun c hc ↦ (h.1 _ hc).2 hb⟩
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      inst✝ : PartialOrder α
      a : α
      h : ∀ (b : α), Membership.mem (Singleton.singleton a) b → LT.lt b a
      ⊢ False
    -/
  · simp at h
    /-
      🎉 no goals
    -/


/-- `{a} ≤ s` in colex iff `s` contains an element greater than or equal to `a`. -/
lemma singleton_le_toColex : (toColex {a} : Colex α) ≤ toColex s ↔ ∃ x ∈ s, a ≤ x := by
  /-
    α : Type u_1
    inst✝ : PartialOrder α
    s : Finset α
    a : α
    ⊢ Iff (LE.le { ofColex := Singleton.singleton a } { ofColex := s }) (Exists fu …
  -/
                                                /-
                                                  🎉 no goals
                                                -/
  simp [toColex_le_toColex]; by_cases a ∈ s <;> aesop
                                                /-
                                                  🎉 no goals
                                                -/


/-- Colex is an extension of the base order. -/
lemma singleton_le_singleton : (toColex {a} : Colex α) ≤ toColex {b} ↔ a ≤ b := by
  /-
    α : Type u_1
    inst✝ : PartialOrder α
    a b : α
    ⊢ Iff (LE.le { ofColex := Singleton.singleton a } { ofColex := Singleton.singl …
  -/
  simp [toColex_le_singleton, eq_comm]
  /-
    🎉 no goals
  -/


/-- Colex is an extension of the base order. -/
lemma singleton_lt_singleton : (toColex {a} : Colex α) < toColex {b} ↔ a < b := by
  /-
    α : Type u_1
    inst✝ : PartialOrder α
    a b : α
    ⊢ Iff (LT.lt { ofColex := Singleton.singleton a } { ofColex := Singleton.singl …
  -/
  simp [toColex_lt_singleton]
  /-
    🎉 no goals
  -/


lemma le_iff_sdiff_subset_lowerClosure {s t : Colex α} :
    s ≤ t ↔ (ofColex s : Set α) \ ofColex t ⊆ lowerClosure (ofColex t \ ofColex s : Set α) := by
  /-
    α : Type u_1
    inst✝ : PartialOrder α
    s t : Finset.Colex α
    ⊢ Iff (LE.le s t) (HasSubset.Subset (SDiff.sdiff ↑s.ofColex ↑t.ofColex) ↑(lowe …
  -/
  simp [le_def, Set.subset_def, and_assoc]
  /-
    🎉 no goals
  -/


instance instDecidableEq : DecidableEq (Colex α) := fun s t ↦
  decidable_of_iff' (s.ofColex = t.ofColex) Colex.ext_iff


instance instDecidableLE [DecidableRel (α := α) (· ≤ ·)] : DecidableRel (α := Colex α) (· ≤ ·) :=
  fun s t ↦ decidable_of_iff'
    (∀ ⦃a⦄, a ∈ ofColex s → a ∉ ofColex t → ∃ b, b ∈ ofColex t ∧ b ∉ ofColex s ∧ a ≤ b) Iff.rfl


instance instDecidableLT [DecidableRel (α := α) (· ≤ ·)] : DecidableRel (α := Colex α) (· < ·) :=
  decidableLTOfDecidableLE


/-- The colexigraphic order is insensitive to removing the same elements from both sets. -/
lemma toColex_sdiff_le_toColex_sdiff (hus : u ⊆ s) (hut : u ⊆ t) :
    toColex (s \ u) ≤ toColex (t \ u) ↔ toColex s ≤ toColex t := by
  simp_rw [toColex_le_toColex, ← and_imp, ← and_assoc, ← mem_sdiff,
    sdiff_sdiff_sdiff_cancel_right (show u ≤ s from hus),
    sdiff_sdiff_sdiff_cancel_right (show u ≤ t from hut)]


/-- The colexigraphic order is insensitive to removing the same elements from both sets. -/
lemma toColex_sdiff_lt_toColex_sdiff (hus : u ⊆ s) (hut : u ⊆ t) :
    toColex (s \ u) < toColex (t \ u) ↔ toColex s < toColex t :=
  lt_iff_lt_of_le_iff_le' (toColex_sdiff_le_toColex_sdiff hut hus) <|
    toColex_sdiff_le_toColex_sdiff hus hut


@[simp] lemma toColex_sdiff_le_toColex_sdiff' :
    toColex (s \ t) ≤ toColex (t \ s) ↔ toColex s ≤ toColex t := by
  /-
    α : Type u_1
    inst✝¹ : PartialOrder α
    s t : Finset α
    inst✝ : DecidableEq α
    ⊢ Iff (LE.le { ofColex := SDiff.sdiff s t } { ofColex := SDiff.sdiff t s }) (L …
  -/
  simpa using toColex_sdiff_le_toColex_sdiff (inter_subset_left (s₁ := s)) inter_subset_right
  /-
    🎉 no goals
  -/


@[simp] lemma toColex_sdiff_lt_toColex_sdiff' :
 toColex (s \ t) < toColex (t \ s) ↔ toColex s < toColex t := by
  /-
    α : Type u_1
    inst✝¹ : PartialOrder α
    s t : Finset α
    inst✝ : DecidableEq α
    ⊢ Iff (LT.lt { ofColex := SDiff.sdiff s t } { ofColex := SDiff.sdiff t s }) (L …
  -/
  simpa using toColex_sdiff_lt_toColex_sdiff (inter_subset_left (s₁ := s)) inter_subset_right
  /-
    🎉 no goals
  -/


@[simp] lemma cons_le_cons (ha hb) : toColex (s.cons a ha) ≤ toColex (s.cons b hb) ↔ a ≤ b := by
  /-
    α : Type u_1
    inst✝ : PartialOrder α
    s : Finset α
    a b : α
    ha : Not (Membership.mem s a)
    hb : Not (Membership.mem s b)
    ⊢ Iff (LE.le { ofColex := Finset.cons a s ha } { ofColex := Finset.cons b s hb …
  -/
  obtain rfl | hab := eq_or_ne a b
    /-
      case inl
      α : Type u_1
      inst✝ : PartialOrder α
      s : Finset α
      a : α
      ha hb : Not (Membership.mem s a)
      ⊢ Iff (LE.le { ofColex := Finset.cons a s ha } { ofColex := Finset.cons a s hb …
    -/
  · simp
    /-
      🎉 no goals
    -/
  classical
  rw [← toColex_sdiff_le_toColex_sdiff', cons_sdiff_cons hab, cons_sdiff_cons hab.symm,
    singleton_le_singleton]


@[simp] lemma cons_lt_cons (ha hb) : toColex (s.cons a ha) < toColex (s.cons b hb) ↔ a < b :=
  lt_iff_lt_of_le_iff_le' (cons_le_cons _ _) (cons_le_cons _ _)


lemma insert_le_insert (ha : a ∉ s) (hb : b ∉ s) :
    toColex (insert a s) ≤ toColex (insert b s) ↔ a ≤ b := by
  /-
    α : Type u_1
    inst✝¹ : PartialOrder α
    s : Finset α
    a b : α
    inst✝ : DecidableEq α
    ha : Not (Membership.mem s a)
    hb : Not (Membership.mem s b)
    ⊢ Iff (LE.le { ofColex := Insert.insert a s } { ofColex := Insert.insert b s } …
  -/
  rw [← cons_eq_insert _ _ ha, ← cons_eq_insert _ _ hb, cons_le_cons]
  /-
    🎉 no goals
  -/


lemma insert_lt_insert (ha : a ∉ s) (hb : b ∉ s) :
    toColex (insert a s) < toColex (insert b s) ↔ a < b := by
  /-
    α : Type u_1
    inst✝¹ : PartialOrder α
    s : Finset α
    a b : α
    inst✝ : DecidableEq α
    ha : Not (Membership.mem s a)
    hb : Not (Membership.mem s b)
    ⊢ Iff (LT.lt { ofColex := Insert.insert a s } { ofColex := Insert.insert b s } …
  -/
  rw [← cons_eq_insert _ _ ha, ← cons_eq_insert _ _ hb, cons_lt_cons]
  /-
    🎉 no goals
  -/


lemma erase_le_erase (ha : a ∈ s) (hb : b ∈ s) :
    toColex (s.erase a) ≤ toColex (s.erase b) ↔ b ≤ a := by
  /-
    α : Type u_1
    inst✝¹ : PartialOrder α
    s : Finset α
    a b : α
    inst✝ : DecidableEq α
    ha : Membership.mem s a
    hb : Membership.mem s b
    ⊢ Iff (LE.le { ofColex := s.erase a } { ofColex := s.erase b }) (LE.le b a)
  -/
  obtain rfl | hab := eq_or_ne a b
    /-
      case inl
      α : Type u_1
      inst✝¹ : PartialOrder α
      s : Finset α
      a : α
      inst✝ : DecidableEq α
      ha hb : Membership.mem s a
      ⊢ Iff (LE.le { ofColex := s.erase a } { ofColex := s.erase a }) (LE.le a a)
    -/
  · simp
    /-
      🎉 no goals
    -/
  classical
  rw [← toColex_sdiff_le_toColex_sdiff', erase_sdiff_erase hab hb, erase_sdiff_erase hab.symm ha,
    singleton_le_singleton]


lemma erase_lt_erase (ha : a ∈ s) (hb : b ∈ s) :
    toColex (s.erase a) < toColex (s.erase b) ↔ b < a :=
  lt_iff_lt_of_le_iff_le' (erase_le_erase hb ha) (erase_le_erase ha hb)


instance instLinearOrder : LinearOrder (Colex α) where
  le_total s t := by
    classical
    obtain rfl | hts := eq_or_ne t s
    · simp
    have ⟨a, ha, hamax⟩ := exists_max_image _ id (symmDiff_nonempty.2 <| ofColex_ne_ofColex.2 hts)
    simp_rw [mem_symmDiff] at ha hamax
    exact ha.imp (fun ha b hbs hbt ↦ ⟨a, ha.1, ha.2, hamax _ <| Or.inr ⟨hbs, hbt⟩⟩)
      (fun ha b hbt hbs ↦ ⟨a, ha.1, ha.2, hamax _ <| Or.inl ⟨hbt, hbs⟩⟩)
  decidableLE := instDecidableLE
  decidableLT := instDecidableLT


private lemma max_mem_aux {s t : Colex α} (hst : s ≠ t) : (ofColex s ∆ ofColex t).Nonempty := by
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    s t : Finset.Colex α
    hst : Ne s t
    ⊢ (symmDiff s.ofColex t.ofColex).Nonempty
  -/
  simpa
  /-
    🎉 no goals
  -/


lemma toColex_lt_toColex_iff_exists_forall_lt :
    toColex s < toColex t ↔ ∃ a ∈ t, a ∉ s ∧ ∀ b ∈ s, b ∉ t → b < a := by
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    s t : Finset α
    ⊢ Iff (LT.lt { ofColex := s } { ofColex := t }) (Exists fun a => And (Membersh …
  -/
  rw [← not_le, toColex_le_toColex, not_forall]
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    s t : Finset α
    ⊢ Iff (Exists fun x => Not (Membership.mem t x → Not (Membership.mem s x) → Ex …
  -/
  simp only [not_forall, not_exists, not_and, not_le, exists_prop, exists_and_left]
  /-
    🎉 no goals
  -/


lemma lt_iff_exists_forall_lt {s t : Colex α} :
    s < t ↔ ∃ a ∈ ofColex t, a ∉ ofColex s ∧ ∀ b ∈ ofColex s, b ∉ ofColex t → b < a :=
  toColex_lt_toColex_iff_exists_forall_lt


lemma toColex_le_toColex_iff_max'_mem :
    toColex s ≤ toColex t ↔ ∀ hst : s ≠ t, (s ∆ t).max' (symmDiff_nonempty.2 hst) ∈ t := by
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    s t : Finset α
    ⊢ Iff (LE.le { ofColex := s } { ofColex := t }) (∀ (hst : Ne s t), Membership. …
  -/
  refine ⟨fun h hst ↦ ?_, fun h a has hat ↦ ?_⟩
    /-
      case refine_1
      α : Type u_1
      inst✝ : LinearOrder α
      s t : Finset α
      h : LE.le { ofColex := s } { ofColex := t }
      hst : Ne s t
      ⊢ Membership.mem t ((symmDiff s t).max' ⋯)
    -/
  · set m := (s ∆ t).max' (symmDiff_nonempty.2 hst)
    /-
      case refine_1
      α : Type u_1
      inst✝ : LinearOrder α
      s t : Finset α
      h : LE.le { ofColex := s } { ofColex := t }
      hst : Ne s t
      m : α := (symmDiff s t).max' ⋯
      ⊢ Membership.mem t m
    -/
    by_contra hmt
    have hms : m ∈ s := by
      simpa [m, mem_symmDiff, hmt] using max'_mem _ <| symmDiff_nonempty.2 hst
    /-
      case refine_1
      α : Type u_1
      inst✝ : LinearOrder α
      s t : Finset α
      h : LE.le { ofColex := s } { ofColex := t }
      hst : Ne s t
      m : α := (symmDiff s t).max' ⋯
      hmt : Not (Membership.mem t m)
      hms : Membership.mem s m
      ⊢ False
    -/
    have ⟨b, hbt, hbs, hmb⟩ := h hms hmt
    exact lt_irrefl _ <| (max'_lt_iff _ _).1 (hmb.lt_of_ne <| ne_of_mem_of_not_mem hms hbs) _ <|
      mem_symmDiff.2 <| Or.inr ⟨hbt, hbs⟩
    /-
      case refine_2
      α : Type u_1
      inst✝ : LinearOrder α
      s t : Finset α
      h : ∀ (hst : Ne s t), Membership.mem t ((symmDiff s t).max' ⋯)
      a : α
      has : Membership.mem { ofColex := s }.ofColex a
      hat : Not (Membership.mem { ofColex := t }.ofColex a)
      ⊢ Exists fun b => And (Membership.mem { ofColex := t }.ofColex b) (And (Not (M …
    -/
  · have hst : s ≠ t := ne_of_mem_of_not_mem' has hat
    /-
      case refine_2
      α : Type u_1
      inst✝ : LinearOrder α
      s t : Finset α
      h : ∀ (hst : Ne s t), Membership.mem t ((symmDiff s t).max' ⋯)
      a : α
      has : Membership.mem { ofColex := s }.ofColex a
      hat : Not (Membership.mem { ofColex := t }.ofColex a)
      hst : Ne s t
      ⊢ Exists fun b => And (Membership.mem { ofColex := t }.ofColex b) (And (Not (M …
    -/
    refine ⟨_, h hst, ?_, le_max' _ _ <| mem_symmDiff.2 <| Or.inl ⟨has, hat⟩⟩
    /-
      case refine_2
      α : Type u_1
      inst✝ : LinearOrder α
      s t : Finset α
      h : ∀ (hst : Ne s t), Membership.mem t ((symmDiff s t).max' ⋯)
      a : α
      has : Membership.mem { ofColex := s }.ofColex a
      hat : Not (Membership.mem { ofColex := t }.ofColex a)
      hst : Ne s t
      ⊢ Not (Membership.mem { ofColex := s }.ofColex ((symmDiff s t).max' ⋯))
    -/
    simpa [mem_symmDiff, h hst] using max'_mem _ <| symmDiff_nonempty.2 hst
    /-
      🎉 no goals
    -/


lemma le_iff_max'_mem {s t : Colex α} :
    s ≤ t ↔ ∀ h : s ≠ t, (ofColex s ∆ ofColex t).max' (max_mem_aux h) ∈ ofColex t :=
  toColex_le_toColex_iff_max'_mem.trans
    ⟨fun h hst ↦ h <| ofColex_ne_ofColex.2 hst, fun h hst ↦ h <| ofColex_ne_ofColex.1 hst⟩


lemma toColex_lt_toColex_iff_max'_mem :
    toColex s < toColex t ↔ ∃ hst : s ≠ t, (s ∆ t).max' (symmDiff_nonempty.2 hst) ∈ t := by
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    s t : Finset α
    ⊢ Iff (LT.lt { ofColex := s } { ofColex := t }) (Exists fun hst => Membership. …
  -/
  rw [lt_iff_le_and_ne, toColex_le_toColex_iff_max'_mem]; aesop
                                                          /-
                                                            🎉 no goals
                                                          -/


lemma lt_iff_max'_mem {s t : Colex α} :
    s < t ↔ ∃ h : s ≠ t, (ofColex s ∆ ofColex t).max' (max_mem_aux h) ∈ ofColex t := by
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    s t : Finset.Colex α
    ⊢ Iff (LT.lt s t) (Exists fun h => Membership.mem t.ofColex ((symmDiff s.ofCol …
  -/
  rw [lt_iff_le_and_ne, le_iff_max'_mem]; aesop
                                          /-
                                            🎉 no goals
                                          -/


lemma lt_iff_exists_filter_lt :
    toColex s < toColex t ↔ ∃ w ∈ t \ s, {a ∈ s | w < a} = {a ∈ t | w < a} := by
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    s t : Finset α
    ⊢ Iff (LT.lt { ofColex := s } { ofColex := t }) (Exists fun w => And (Membersh …
  -/
  simp only [lt_iff_exists_forall_lt, mem_sdiff, filter_inj, and_assoc]
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    s t : Finset α
    ⊢ Iff (Exists fun a => And (Membership.mem t a) (And (Not (Membership.mem s a) …
  -/
  refine ⟨fun h ↦ ?_, ?_⟩
    /-
      case refine_1
      α : Type u_1
      inst✝ : LinearOrder α
      s t : Finset α
      h : Exists fun a => And (Membership.mem t a) (And (Not (Membership.mem s a)) ( …
      ⊢ Exists fun w => And (Membership.mem t w) (And (Not (Membership.mem s w)) (∀  …
    -/
  · let u := {w ∈ t \ s | ∀ a ∈ s, a ∉ t → a < w}
    /-
      case refine_1
      α : Type u_1
      inst✝ : LinearOrder α
      s t : Finset α
      h : Exists fun a => And (Membership.mem t a) (And (Not (Membership.mem s a)) ( …
      u : Finset α := Finset.filter (fun w => ∀ (a : α), Membership.mem s a → Not (M …
      ⊢ Exists fun w => And (Membership.mem t w) (And (Not (Membership.mem s w)) (∀  …
    -/
    have mem_u {w : α} : w ∈ u ↔ w ∈ t ∧ w ∉ s ∧ ∀ a ∈ s, a ∉ t → a < w := by simp [u, and_assoc]
    /-
      case refine_1
      α : Type u_1
      inst✝ : LinearOrder α
      s t : Finset α
      h : Exists fun a => And (Membership.mem t a) (And (Not (Membership.mem s a)) ( …
      u : Finset α := Finset.filter (fun w => ∀ (a : α), Membership.mem s a → Not (M …
      mem_u : ∀ {w : α}, Iff (Membership.mem u w) (And (Membership.mem t w) (And (No …
      ⊢ Exists fun w => And (Membership.mem t w) (And (Not (Membership.mem s w)) (∀  …
    -/
    have hu : u.Nonempty := h.imp fun _ ↦ mem_u.2
    /-
      case refine_1
      α : Type u_1
      inst✝ : LinearOrder α
      s t : Finset α
      h : Exists fun a => And (Membership.mem t a) (And (Not (Membership.mem s a)) ( …
      u : Finset α := Finset.filter (fun w => ∀ (a : α), Membership.mem s a → Not (M …
      mem_u : ∀ {w : α}, Iff (Membership.mem u w) (And (Membership.mem t w) (And (No …
      hu : u.Nonempty
      ⊢ Exists fun w => And (Membership.mem t w) (And (Not (Membership.mem s w)) (∀  …
    -/
    let m := max' _ hu
    /-
      case refine_1
      α : Type u_1
      inst✝ : LinearOrder α
      s t : Finset α
      h : Exists fun a => And (Membership.mem t a) (And (Not (Membership.mem s a)) ( …
      u : Finset α := Finset.filter (fun w => ∀ (a : α), Membership.mem s a → Not (M …
      mem_u : ∀ {w : α}, Iff (Membership.mem u w) (And (Membership.mem t w) (And (No …
      hu : u.Nonempty
      m : α := u.max' hu
      ⊢ Exists fun w => And (Membership.mem t w) (And (Not (Membership.mem s w)) (∀  …
    -/
    have ⟨hmt, hms, hm⟩ : m ∈ t ∧ m ∉ s ∧ ∀ a ∈ s, a ∉ t → a < m := mem_u.1 <| max'_mem _ _
    /-
      case refine_1
      α : Type u_1
      inst✝ : LinearOrder α
      s t : Finset α
      h : Exists fun a => And (Membership.mem t a) (And (Not (Membership.mem s a)) ( …
      u : Finset α := Finset.filter (fun w => ∀ (a : α), Membership.mem s a → Not (M …
      mem_u : ∀ {w : α}, Iff (Membership.mem u w) (And (Membership.mem t w) (And (No …
      hu : u.Nonempty
      m : α := u.max' hu
      hmt : Membership.mem t m
      hms : Not (Membership.mem s m)
      hm : ∀ (a : α), Membership.mem s a → Not (Membership.mem t a) → LT.lt a m
      ⊢ Exists fun w => And (Membership.mem t w) (And (Not (Membership.mem s w)) (∀  …
    -/
    refine ⟨m, hmt, hms, fun a hma ↦ ⟨fun has ↦ not_imp_comm.1 (hm _ has) hma.asymm, fun hat ↦ ?_⟩⟩
    /-
      case refine_1
      α : Type u_1
      inst✝ : LinearOrder α
      s t : Finset α
      h : Exists fun a => And (Membership.mem t a) (And (Not (Membership.mem s a)) ( …
      u : Finset α := Finset.filter (fun w => ∀ (a : α), Membership.mem s a → Not (M …
      mem_u : ∀ {w : α}, Iff (Membership.mem u w) (And (Membership.mem t w) (And (No …
      hu : u.Nonempty
      m : α := u.max' hu
      hmt : Membership.mem t m
      hms : Not (Membership.mem s m)
      hm : ∀ (a : α), Membership.mem s a → Not (Membership.mem t a) → LT.lt a m
      a : α
      hma : LT.lt m a
      hat : Membership.mem t a
      ⊢ Membership.mem s a
    -/
    by_contra has
    /-
      case refine_1
      α : Type u_1
      inst✝ : LinearOrder α
      s t : Finset α
      h : Exists fun a => And (Membership.mem t a) (And (Not (Membership.mem s a)) ( …
      u : Finset α := Finset.filter (fun w => ∀ (a : α), Membership.mem s a → Not (M …
      mem_u : ∀ {w : α}, Iff (Membership.mem u w) (And (Membership.mem t w) (And (No …
      hu : u.Nonempty
      m : α := u.max' hu
      hmt : Membership.mem t m
      hms : Not (Membership.mem s m)
      hm : ∀ (a : α), Membership.mem s a → Not (Membership.mem t a) → LT.lt a m
      a : α
      hma : LT.lt m a
      hat : Membership.mem t a
      has : Not (Membership.mem s a)
      ⊢ False
    -/
    have hau : a ∈ u := mem_u.2 ⟨hat, has, fun b hbs hbt ↦ (hm _ hbs hbt).trans hma⟩
    /-
      case refine_1
      α : Type u_1
      inst✝ : LinearOrder α
      s t : Finset α
      h : Exists fun a => And (Membership.mem t a) (And (Not (Membership.mem s a)) ( …
      u : Finset α := Finset.filter (fun w => ∀ (a : α), Membership.mem s a → Not (M …
      mem_u : ∀ {w : α}, Iff (Membership.mem u w) (And (Membership.mem t w) (And (No …
      hu : u.Nonempty
      m : α := u.max' hu
      hmt : Membership.mem t m
      hms : Not (Membership.mem s m)
      hm : ∀ (a : α), Membership.mem s a → Not (Membership.mem t a) → LT.lt a m
      a : α
      hma : LT.lt m a
      hat : Membership.mem t a
      has : Not (Membership.mem s a)
      hau : Membership.mem u a
      ⊢ False
    -/
    exact hma.not_le <| le_max' _ _ hau
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      inst✝ : LinearOrder α
      s t : Finset α
      ⊢ (Exists fun w => And (Membership.mem t w) (And (Not (Membership.mem s w)) (∀ …
    -/
  · rintro ⟨w, hwt, hws, hw⟩
    /-
      case refine_2.intro.intro.intro
      α : Type u_1
      inst✝ : LinearOrder α
      s t : Finset α
      w : α
      hwt : Membership.mem t w
      hws : Not (Membership.mem s w)
      hw : ∀ ⦃a : α⦄, LT.lt w a → Iff (Membership.mem s a) (Membership.mem t a)
      ⊢ Exists fun a => And (Membership.mem t a) (And (Not (Membership.mem s a)) (∀  …
    -/
    refine ⟨w, hwt, hws, fun a has hat ↦ ?_⟩
    /-
      case refine_2.intro.intro.intro
      α : Type u_1
      inst✝ : LinearOrder α
      s t : Finset α
      w : α
      hwt : Membership.mem t w
      hws : Not (Membership.mem s w)
      hw : ∀ ⦃a : α⦄, LT.lt w a → Iff (Membership.mem s a) (Membership.mem t a)
      a : α
      has : Membership.mem s a
      hat : Not (Membership.mem t a)
      ⊢ LT.lt a w
    -/
    by_contra! hwa
    /-
      case refine_2.intro.intro.intro
      α : Type u_1
      inst✝ : LinearOrder α
      s t : Finset α
      w : α
      hwt : Membership.mem t w
      hws : Not (Membership.mem s w)
      hw : ∀ ⦃a : α⦄, LT.lt w a → Iff (Membership.mem s a) (Membership.mem t a)
      a : α
      has : Membership.mem s a
      hat : Not (Membership.mem t a)
      hwa : LE.le w a
      ⊢ False
    -/
    exact hat <| (hw <| hwa.lt_of_ne <| ne_of_mem_of_not_mem hwt hat).1 has
    /-
      🎉 no goals
    -/


/-- If `s ≤ t` in colex and `#s ≤ #t`, then `s \ {a} ≤ t \ {min t}` for any `a ∈ s`. -/
lemma erase_le_erase_min' (hst : toColex s ≤ toColex t) (hcard : #s ≤ #t) (ha : a ∈ s) :
    toColex (s.erase a) ≤
      toColex (t.erase <| min' t <| card_pos.1 <| (card_pos.2 ⟨a, ha⟩).trans_le hcard) := by
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    s t : Finset α
    a : α
    hst : LE.le { ofColex := s } { ofColex := t }
    hcard : LE.le s.card t.card
    ha : Membership.mem s a
    ⊢ LE.le { ofColex := s.erase a } { ofColex := t.erase (t.min' ⋯) }
  -/
  generalize_proofs ht
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    s t : Finset α
    a : α
    hst : LE.le { ofColex := s } { ofColex := t }
    hcard : LE.le s.card t.card
    ha : Membership.mem s a
    ht : t.Nonempty
    ⊢ LE.le { ofColex := s.erase a } { ofColex := t.erase (t.min' ht) }
  -/
  set m := min' t ht
  -- Case on whether `s = t`
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    s t : Finset α
    a : α
    hst : LE.le { ofColex := s } { ofColex := t }
    hcard : LE.le s.card t.card
    ha : Membership.mem s a
    ht : t.Nonempty
    m : α := t.min' ht
    ⊢ LE.le { ofColex := s.erase a } { ofColex := t.erase m }
  -/
  obtain rfl | h' := eq_or_ne s t
  -- If `s = t`, then `s \ {a} ≤ s \ {m}` because `m ≤ a`
    /-
      case inl
      α : Type u_1
      inst✝ : LinearOrder α
      s : Finset α
      a : α
      ha : Membership.mem s a
      hst : LE.le { ofColex := s } { ofColex := s }
      hcard : LE.le s.card s.card
      ht : s.Nonempty
      m : α := s.min' ht
      ⊢ LE.le { ofColex := s.erase a } { ofColex := s.erase m }
    -/
  · exact (erase_le_erase ha <| min'_mem _ _).2 <| min'_le _ _ <| ha
    /-
      🎉 no goals
    -/
  -- If `s ≠ t`, call `w` the colex witness. Case on whether `w < a` or `a < w`
  /-
    case inr
    α : Type u_1
    inst✝ : LinearOrder α
    s t : Finset α
    a : α
    hst : LE.le { ofColex := s } { ofColex := t }
    hcard : LE.le s.card t.card
    ha : Membership.mem s a
    ht : t.Nonempty
    m : α := t.min' ht
    h' : Ne s t
    ⊢ LE.le { ofColex := s.erase a } { ofColex := t.erase m }
  -/
  replace hst := hst.lt_of_ne <| toColex_inj.not.2 h'
  /-
    case inr
    α : Type u_1
    inst✝ : LinearOrder α
    s t : Finset α
    a : α
    hcard : LE.le s.card t.card
    ha : Membership.mem s a
    ht : t.Nonempty
    m : α := t.min' ht
    h' : Ne s t
    hst : LT.lt { ofColex := s } { ofColex := t }
    ⊢ LE.le { ofColex := s.erase a } { ofColex := t.erase m }
  -/
  simp only [lt_iff_exists_filter_lt, mem_sdiff, filter_inj, and_assoc] at hst
  /-
    case inr
    α : Type u_1
    inst✝ : LinearOrder α
    s t : Finset α
    a : α
    hcard : LE.le s.card t.card
    ha : Membership.mem s a
    ht : t.Nonempty
    m : α := t.min' ht
    h' : Ne s t
    hst : Exists fun w => And (Membership.mem t w) (And (Not (Membership.mem s w)) …
    ⊢ LE.le { ofColex := s.erase a } { ofColex := t.erase m }
  -/
  obtain ⟨w, hwt, hws, hw⟩ := hst
  /-
    case inr.intro.intro.intro
    α : Type u_1
    inst✝ : LinearOrder α
    s t : Finset α
    a : α
    hcard : LE.le s.card t.card
    ha : Membership.mem s a
    ht : t.Nonempty
    m : α := t.min' ht
    h' : Ne s t
    w : α
    hwt : Membership.mem t w
    hws : Not (Membership.mem s w)
    hw : ∀ ⦃a : α⦄, LT.lt w a → Iff (Membership.mem s a) (Membership.mem t a)
    ⊢ LE.le { ofColex := s.erase a } { ofColex := t.erase m }
  -/
  obtain hwa | haw := (ne_of_mem_of_not_mem ha hws).symm.lt_or_lt
  -- If `w < a`, then `a` is the colex witness for `s \ {a} < t \ {m}`
    /-
      case inr.intro.intro.intro.inl
      α : Type u_1
      inst✝ : LinearOrder α
      s t : Finset α
      a : α
      hcard : LE.le s.card t.card
      ha : Membership.mem s a
      ht : t.Nonempty
      m : α := t.min' ht
      h' : Ne s t
      w : α
      hwt : Membership.mem t w
      hws : Not (Membership.mem s w)
      hw : ∀ ⦃a : α⦄, LT.lt w a → Iff (Membership.mem s a) (Membership.mem t a)
      hwa : LT.lt w a
      ⊢ LE.le { ofColex := s.erase a } { ofColex := t.erase m }
    -/
  · have hma : m < a := (min'_le _ _ hwt).trans_lt hwa
    refine (lt_iff_exists_forall_lt.2 ⟨a, mem_erase.2 ⟨hma.ne', (hw hwa).1 ha⟩,
      not_mem_erase _ _, fun b hbs hbt ↦ ?_⟩).le
    /-
      case inr.intro.intro.intro.inl
      α : Type u_1
      inst✝ : LinearOrder α
      s t : Finset α
      a : α
      hcard : LE.le s.card t.card
      ha : Membership.mem s a
      ht : t.Nonempty
      m : α := t.min' ht
      h' : Ne s t
      w : α
      hwt : Membership.mem t w
      hws : Not (Membership.mem s w)
      hw : ∀ ⦃a : α⦄, LT.lt w a → Iff (Membership.mem s a) (Membership.mem t a)
      hwa : LT.lt w a
      hma : LT.lt m a
      b : α
      hbs : Membership.mem { ofColex := { val := s.val.erase a, nodup := ⋯ } }.ofCol …
      hbt : Not (Membership.mem { ofColex := { val := t.val.erase m, nodup := ⋯ } }. …
      ⊢ LT.lt b a
    -/
    change b ∉ t.erase m at hbt
    /-
      case inr.intro.intro.intro.inl
      α : Type u_1
      inst✝ : LinearOrder α
      s t : Finset α
      a : α
      hcard : LE.le s.card t.card
      ha : Membership.mem s a
      ht : t.Nonempty
      m : α := t.min' ht
      h' : Ne s t
      w : α
      hwt : Membership.mem t w
      hws : Not (Membership.mem s w)
      hw : ∀ ⦃a : α⦄, LT.lt w a → Iff (Membership.mem s a) (Membership.mem t a)
      hwa : LT.lt w a
      hma : LT.lt m a
      b : α
      hbs : Membership.mem { ofColex := { val := s.val.erase a, nodup := ⋯ } }.ofCol …
      hbt : Not (Membership.mem (t.erase m) b)
      ⊢ LT.lt b a
    -/
    rw [mem_erase, not_and_or, not_ne_iff] at hbt
    /-
      case inr.intro.intro.intro.inl
      α : Type u_1
      inst✝ : LinearOrder α
      s t : Finset α
      a : α
      hcard : LE.le s.card t.card
      ha : Membership.mem s a
      ht : t.Nonempty
      m : α := t.min' ht
      h' : Ne s t
      w : α
      hwt : Membership.mem t w
      hws : Not (Membership.mem s w)
      hw : ∀ ⦃a : α⦄, LT.lt w a → Iff (Membership.mem s a) (Membership.mem t a)
      hwa : LT.lt w a
      hma : LT.lt m a
      b : α
      hbs : Membership.mem { ofColex := { val := s.val.erase a, nodup := ⋯ } }.ofCol …
      hbt : Or (Eq b m) (Not (Membership.mem t b))
      ⊢ LT.lt b a
    -/
    obtain rfl | hbt := hbt
      /-
        case inr.intro.intro.intro.inl.inl
        α : Type u_1
        inst✝ : LinearOrder α
        s t : Finset α
        a : α
        hcard : LE.le s.card t.card
        ha : Membership.mem s a
        ht : t.Nonempty
        m : α := t.min' ht
        h' : Ne s t
        w : α
        hwt : Membership.mem t w
        hws : Not (Membership.mem s w)
        hw : ∀ ⦃a : α⦄, LT.lt w a → Iff (Membership.mem s a) (Membership.mem t a)
        hwa : LT.lt w a
        hma : LT.lt m a
        hbs : Membership.mem { ofColex := { val := s.val.erase a, nodup := ⋯ } }.ofCol …
        ⊢ LT.lt m a
      -/
    · assumption
      /-
        🎉 no goals
      -/
      /-
        case inr.intro.intro.intro.inl.inr
        α : Type u_1
        inst✝ : LinearOrder α
        s t : Finset α
        a : α
        hcard : LE.le s.card t.card
        ha : Membership.mem s a
        ht : t.Nonempty
        m : α := t.min' ht
        h' : Ne s t
        w : α
        hwt : Membership.mem t w
        hws : Not (Membership.mem s w)
        hw : ∀ ⦃a : α⦄, LT.lt w a → Iff (Membership.mem s a) (Membership.mem t a)
        hwa : LT.lt w a
        hma : LT.lt m a
        b : α
        hbs : Membership.mem { ofColex := { val := s.val.erase a, nodup := ⋯ } }.ofCol …
        hbt : Not (Membership.mem t b)
        ⊢ LT.lt b a
      -/
    · by_contra! hab
      /-
        case inr.intro.intro.intro.inl.inr
        α : Type u_1
        inst✝ : LinearOrder α
        s t : Finset α
        a : α
        hcard : LE.le s.card t.card
        ha : Membership.mem s a
        ht : t.Nonempty
        m : α := t.min' ht
        h' : Ne s t
        w : α
        hwt : Membership.mem t w
        hws : Not (Membership.mem s w)
        hw : ∀ ⦃a : α⦄, LT.lt w a → Iff (Membership.mem s a) (Membership.mem t a)
        hwa : LT.lt w a
        hma : LT.lt m a
        b : α
        hbs : Membership.mem { ofColex := { val := s.val.erase a, nodup := ⋯ } }.ofCol …
        hbt : Not (Membership.mem t b)
        hab : LE.le a b
        ⊢ False
      -/
      exact hbt <| (hw <| hwa.trans_le hab).1 <| mem_of_mem_erase hbs
      /-
        🎉 no goals
      -/
  -- If `a < w`, case on whether `m < w` or `m = w`
  /-
    case inr.intro.intro.intro.inr
    α : Type u_1
    inst✝ : LinearOrder α
    s t : Finset α
    a : α
    hcard : LE.le s.card t.card
    ha : Membership.mem s a
    ht : t.Nonempty
    m : α := t.min' ht
    h' : Ne s t
    w : α
    hwt : Membership.mem t w
    hws : Not (Membership.mem s w)
    hw : ∀ ⦃a : α⦄, LT.lt w a → Iff (Membership.mem s a) (Membership.mem t a)
    haw : LT.lt a w
    ⊢ LE.le { ofColex := s.erase a } { ofColex := t.erase m }
  -/
  obtain rfl | hmw : m = w ∨ m < w := (min'_le _ _ hwt).eq_or_lt
  -- If `m = w`, then `s \ {a} = t \ {m}`
  · have : erase t m ⊆ erase s a := by
      rintro b hb
      rw [mem_erase] at hb ⊢
      exact ⟨(haw.trans_le <| min'_le _ _ hb.2).ne',
        (hw <| hb.1.lt_of_le' <| min'_le _ _ hb.2).2 hb.2⟩
    /-
      case inr.intro.intro.intro.inr.inl
      α : Type u_1
      inst✝ : LinearOrder α
      s t : Finset α
      a : α
      hcard : LE.le s.card t.card
      ha : Membership.mem s a
      ht : t.Nonempty
      m : α := t.min' ht
      h' : Ne s t
      hwt : Membership.mem t m
      hws : Not (Membership.mem s m)
      hw : ∀ ⦃a : α⦄, LT.lt m a → Iff (Membership.mem s a) (Membership.mem t a)
      haw : LT.lt a m
      this : HasSubset.Subset (t.erase m) (s.erase a)
      ⊢ LE.le { ofColex := s.erase a } { ofColex := t.erase m }
    -/
    rw [eq_of_subset_of_card_le this]
    /-
      case inr.intro.intro.intro.inr.inl
      α : Type u_1
      inst✝ : LinearOrder α
      s t : Finset α
      a : α
      hcard : LE.le s.card t.card
      ha : Membership.mem s a
      ht : t.Nonempty
      m : α := t.min' ht
      h' : Ne s t
      hwt : Membership.mem t m
      hws : Not (Membership.mem s m)
      hw : ∀ ⦃a : α⦄, LT.lt m a → Iff (Membership.mem s a) (Membership.mem t a)
      haw : LT.lt a m
      this : HasSubset.Subset (t.erase m) (s.erase a)
      ⊢ LE.le (s.erase a).card (t.erase m).card
    -/
    rw [card_erase_of_mem ha, card_erase_of_mem (min'_mem _ _)]
    /-
      case inr.intro.intro.intro.inr.inl
      α : Type u_1
      inst✝ : LinearOrder α
      s t : Finset α
      a : α
      hcard : LE.le s.card t.card
      ha : Membership.mem s a
      ht : t.Nonempty
      m : α := t.min' ht
      h' : Ne s t
      hwt : Membership.mem t m
      hws : Not (Membership.mem s m)
      hw : ∀ ⦃a : α⦄, LT.lt m a → Iff (Membership.mem s a) (Membership.mem t a)
      haw : LT.lt a m
      this : HasSubset.Subset (t.erase m) (s.erase a)
      ⊢ LE.le (HSub.hSub s.card 1) (HSub.hSub t.card 1)
    -/
    exact tsub_le_tsub_right hcard _
    /-
      🎉 no goals
    -/
  -- If `m < w`, then `w` works as the colex witness for  `s \ {a} < t \ {m}`
  · refine (lt_iff_exists_forall_lt.2 ⟨w, mem_erase.2 ⟨hmw.ne', hwt⟩, mt mem_of_mem_erase hws,
      fun b hbs hbt ↦ ?_⟩).le
    /-
      case inr.intro.intro.intro.inr.inr
      α : Type u_1
      inst✝ : LinearOrder α
      s t : Finset α
      a : α
      hcard : LE.le s.card t.card
      ha : Membership.mem s a
      ht : t.Nonempty
      m : α := t.min' ht
      h' : Ne s t
      w : α
      hwt : Membership.mem t w
      hws : Not (Membership.mem s w)
      hw : ∀ ⦃a : α⦄, LT.lt w a → Iff (Membership.mem s a) (Membership.mem t a)
      haw : LT.lt a w
      hmw : LT.lt m w
      b : α
      hbs : Membership.mem { ofColex := { val := s.val.erase a, nodup := ⋯ } }.ofCol …
      hbt : Not (Membership.mem { ofColex := { val := t.val.erase m, nodup := ⋯ } }. …
      ⊢ LT.lt b w
    -/
    change b ∉ t.erase m at hbt
    /-
      case inr.intro.intro.intro.inr.inr
      α : Type u_1
      inst✝ : LinearOrder α
      s t : Finset α
      a : α
      hcard : LE.le s.card t.card
      ha : Membership.mem s a
      ht : t.Nonempty
      m : α := t.min' ht
      h' : Ne s t
      w : α
      hwt : Membership.mem t w
      hws : Not (Membership.mem s w)
      hw : ∀ ⦃a : α⦄, LT.lt w a → Iff (Membership.mem s a) (Membership.mem t a)
      haw : LT.lt a w
      hmw : LT.lt m w
      b : α
      hbs : Membership.mem { ofColex := { val := s.val.erase a, nodup := ⋯ } }.ofCol …
      hbt : Not (Membership.mem (t.erase m) b)
      ⊢ LT.lt b w
    -/
    rw [mem_erase, not_and_or, not_ne_iff] at hbt
    /-
      case inr.intro.intro.intro.inr.inr
      α : Type u_1
      inst✝ : LinearOrder α
      s t : Finset α
      a : α
      hcard : LE.le s.card t.card
      ha : Membership.mem s a
      ht : t.Nonempty
      m : α := t.min' ht
      h' : Ne s t
      w : α
      hwt : Membership.mem t w
      hws : Not (Membership.mem s w)
      hw : ∀ ⦃a : α⦄, LT.lt w a → Iff (Membership.mem s a) (Membership.mem t a)
      haw : LT.lt a w
      hmw : LT.lt m w
      b : α
      hbs : Membership.mem { ofColex := { val := s.val.erase a, nodup := ⋯ } }.ofCol …
      hbt : Or (Eq b m) (Not (Membership.mem t b))
      ⊢ LT.lt b w
    -/
    obtain rfl | hbt := hbt
      /-
        case inr.intro.intro.intro.inr.inr.inl
        α : Type u_1
        inst✝ : LinearOrder α
        s t : Finset α
        a : α
        hcard : LE.le s.card t.card
        ha : Membership.mem s a
        ht : t.Nonempty
        m : α := t.min' ht
        h' : Ne s t
        w : α
        hwt : Membership.mem t w
        hws : Not (Membership.mem s w)
        hw : ∀ ⦃a : α⦄, LT.lt w a → Iff (Membership.mem s a) (Membership.mem t a)
        haw : LT.lt a w
        hmw : LT.lt m w
        hbs : Membership.mem { ofColex := { val := s.val.erase a, nodup := ⋯ } }.ofCol …
        ⊢ LT.lt m w
      -/
    · assumption
      /-
        🎉 no goals
      -/
      /-
        case inr.intro.intro.intro.inr.inr.inr
        α : Type u_1
        inst✝ : LinearOrder α
        s t : Finset α
        a : α
        hcard : LE.le s.card t.card
        ha : Membership.mem s a
        ht : t.Nonempty
        m : α := t.min' ht
        h' : Ne s t
        w : α
        hwt : Membership.mem t w
        hws : Not (Membership.mem s w)
        hw : ∀ ⦃a : α⦄, LT.lt w a → Iff (Membership.mem s a) (Membership.mem t a)
        haw : LT.lt a w
        hmw : LT.lt m w
        b : α
        hbs : Membership.mem { ofColex := { val := s.val.erase a, nodup := ⋯ } }.ofCol …
        hbt : Not (Membership.mem t b)
        ⊢ LT.lt b w
      -/
    · by_contra! hwb
      /-
        case inr.intro.intro.intro.inr.inr.inr
        α : Type u_1
        inst✝ : LinearOrder α
        s t : Finset α
        a : α
        hcard : LE.le s.card t.card
        ha : Membership.mem s a
        ht : t.Nonempty
        m : α := t.min' ht
        h' : Ne s t
        w : α
        hwt : Membership.mem t w
        hws : Not (Membership.mem s w)
        hw : ∀ ⦃a : α⦄, LT.lt w a → Iff (Membership.mem s a) (Membership.mem t a)
        haw : LT.lt a w
        hmw : LT.lt m w
        b : α
        hbs : Membership.mem { ofColex := { val := s.val.erase a, nodup := ⋯ } }.ofCol …
        hbt : Not (Membership.mem t b)
        hwb : LE.le w b
        ⊢ False
      -/
      exact hbt <| (hw <| hwb.lt_of_ne <| ne_of_mem_of_not_mem hwt hbt).1 <| mem_of_mem_erase hbs
      /-
        🎉 no goals
      -/


/-- Strictly monotone functions preserve the colex ordering. -/
lemma toColex_image_le_toColex_image (hf : StrictMono f) :
    toColex (s.image f) ≤ toColex (t.image f) ↔ toColex s ≤ toColex t := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : LinearOrder α
    inst✝ : LinearOrder β
    f : α → β
    s t : Finset α
    hf : StrictMono f
    ⊢ Iff (LE.le { ofColex := Finset.image f s } { ofColex := Finset.image f t })  …
  -/
  simp [toColex_le_toColex, hf.le_iff_le, hf.injective.eq_iff]
  /-
    🎉 no goals
  -/


/-- Strictly monotone functions preserve the colex ordering. -/
lemma toColex_image_lt_toColex_image (hf : StrictMono f) :
    toColex (s.image f) < toColex (t.image f) ↔ toColex s < toColex t :=
  lt_iff_lt_of_le_iff_le <| toColex_image_le_toColex_image hf


lemma toColex_image_ofColex_strictMono (hf : StrictMono f) :
    StrictMono fun s ↦ toColex <| image f <| ofColex s :=
  fun _s _t ↦ (toColex_image_lt_toColex_image hf).2


instance instBoundedOrder : BoundedOrder (Colex α) where
  top := toColex univ
  le_top _x := toColex_le_toColex_of_subset <| subset_univ _


@[simp] lemma toColex_univ : toColex (univ : Finset α) = ⊤ := rfl

@[simp] lemma ofColex_top : ofColex (⊤ : Colex α) = univ := rfl


/-- `𝒜` is an initial segment of the colexigraphic order on sets of `r`, and that if `t` is below
`s` in colex where `t` has size `r` and `s` is in `𝒜`, then `t` is also in `𝒜`. In effect, `𝒜` is
downwards closed with respect to colex among sets of size `r`. -/
def IsInitSeg (𝒜 : Finset (Finset α)) (r : ℕ) : Prop :=
  (𝒜 : Set (Finset α)).Sized r ∧
    ∀ ⦃s t : Finset α⦄, s ∈ 𝒜 → toColex t < toColex s ∧ #t = r → t ∈ 𝒜


                                                                          /-
                                                                            α : Type u_1
                                                                            inst✝ : LinearOrder α
                                                                            r : Nat
                                                                            ⊢ Finset.Colex.IsInitSeg EmptyCollection.emptyCollection r
                                                                          -/
@[simp] lemma isInitSeg_empty : IsInitSeg (∅ : Finset (Finset α)) r := by simp [IsInitSeg]
                                                                          /-
                                                                            🎉 no goals
                                                                          -/


/-- Initial segments are nested in some way. In particular, if they're the same size they're equal.
-/
lemma IsInitSeg.total (h₁ : IsInitSeg 𝒜₁ r) (h₂ : IsInitSeg 𝒜₂ r) : 𝒜₁ ⊆ 𝒜₂ ∨ 𝒜₂ ⊆ 𝒜₁ := by
  classical
  simp_rw [← sdiff_eq_empty_iff_subset, ← not_nonempty_iff_eq_empty]
  by_contra! h
  have ⟨⟨s, hs⟩, t, ht⟩ := h
  rw [mem_sdiff] at hs ht
  obtain hst | hst | hts := trichotomous_of (α := Colex α) (· < ·) (toColex s) (toColex t)
  · exact hs.2 <| h₂.2 ht.1 ⟨hst, h₁.1 hs.1⟩
  · simp only [toColex.injEq] at hst
    exact ht.2 <| hst ▸ hs.1
  · exact ht.2 <| h₁.2 hs.1 ⟨hts, h₂.1 ht.1⟩


/-- The initial segment of the colexicographic order on sets with `#s` elements and ending at
`s`. -/
def initSeg (s : Finset α) : Finset (Finset α) := {t | #s = #t ∧ toColex t ≤ toColex s}


@[simp]
                                                                          /-
                                                                            α : Type u_1
                                                                            inst✝¹ : LinearOrder α
                                                                            s t : Finset α
                                                                            inst✝ : Fintype α
                                                                            ⊢ Iff (Membership.mem (Finset.Colex.initSeg s) t) (And (Eq s.card t.card) (LE. …
                                                                          -/
lemma mem_initSeg : t ∈ initSeg s ↔ #s = #t ∧ toColex t ≤ toColex s := by simp [initSeg]
                                                                          /-
                                                                            🎉 no goals
                                                                          -/


                                             /-
                                               α : Type u_1
                                               inst✝¹ : LinearOrder α
                                               s : Finset α
                                               inst✝ : Fintype α
                                               ⊢ Membership.mem (Finset.Colex.initSeg s) s
                                             -/
lemma mem_initSeg_self : s ∈ initSeg s := by simp
                                             /-
                                               🎉 no goals
                                             -/

@[simp] lemma initSeg_nonempty : (initSeg s).Nonempty := ⟨s, mem_initSeg_self⟩


lemma isInitSeg_initSeg : IsInitSeg (initSeg s) #s := by
  /-
    α : Type u_1
    inst✝¹ : LinearOrder α
    s : Finset α
    inst✝ : Fintype α
    ⊢ Finset.Colex.IsInitSeg (Finset.Colex.initSeg s) s.card
  -/
  refine ⟨fun t ht => (mem_initSeg.1 ht).1.symm, fun t₁ t₂ ht₁ ht₂ ↦ mem_initSeg.2 ⟨ht₂.2.symm, ?_⟩⟩
  /-
    α : Type u_1
    inst✝¹ : LinearOrder α
    s : Finset α
    inst✝ : Fintype α
    t₁ t₂ : Finset α
    ht₁ : Membership.mem (Finset.Colex.initSeg s) t₁
    ht₂ : And (LT.lt { ofColex := t₂ } { ofColex := t₁ }) (Eq t₂.card s.card)
    ⊢ LE.le { ofColex := t₂ } { ofColex := s }
  -/
  rw [mem_initSeg] at ht₁
  /-
    α : Type u_1
    inst✝¹ : LinearOrder α
    s : Finset α
    inst✝ : Fintype α
    t₁ t₂ : Finset α
    ht₁ : And (Eq s.card t₁.card) (LE.le { ofColex := t₁ } { ofColex := s })
    ht₂ : And (LT.lt { ofColex := t₂ } { ofColex := t₁ }) (Eq t₂.card s.card)
    ⊢ LE.le { ofColex := t₂ } { ofColex := s }
  -/
  exact ht₂.1.le.trans ht₁.2
  /-
    🎉 no goals
  -/


lemma IsInitSeg.exists_initSeg (h𝒜 : IsInitSeg 𝒜 r) (h𝒜₀ : 𝒜.Nonempty) :
    ∃ s : Finset α, #s = r ∧ 𝒜 = initSeg s := by
  have hs := sup'_mem (ofColex ⁻¹' 𝒜) (LinearOrder.supClosed _) 𝒜 h𝒜₀ toColex
    (fun a ha ↦ by simpa using ha)
  /-
    α : Type u_1
    inst✝¹ : LinearOrder α
    𝒜 : Finset (Finset α)
    r : Nat
    inst✝ : Fintype α
    h𝒜 : Finset.Colex.IsInitSeg 𝒜 r
    h𝒜₀ : 𝒜.Nonempty
    hs : Membership.mem (Set.preimage Finset.Colex.ofColex ↑𝒜) (𝒜.sup' h𝒜₀ Finset. …
    ⊢ Exists fun s => And (Eq s.card r) (Eq 𝒜 (Finset.Colex.initSeg s))
  -/
  refine ⟨_, h𝒜.1 hs, ?_⟩
  /-
    α : Type u_1
    inst✝¹ : LinearOrder α
    𝒜 : Finset (Finset α)
    r : Nat
    inst✝ : Fintype α
    h𝒜 : Finset.Colex.IsInitSeg 𝒜 r
    h𝒜₀ : 𝒜.Nonempty
    hs : Membership.mem (Set.preimage Finset.Colex.ofColex ↑𝒜) (𝒜.sup' h𝒜₀ Finset. …
    ⊢ Eq 𝒜 (Finset.Colex.initSeg (𝒜.sup' h𝒜₀ Finset.Colex.toColex).ofColex)
  -/
  ext t
  /-
    case h
    α : Type u_1
    inst✝¹ : LinearOrder α
    𝒜 : Finset (Finset α)
    r : Nat
    inst✝ : Fintype α
    h𝒜 : Finset.Colex.IsInitSeg 𝒜 r
    h𝒜₀ : 𝒜.Nonempty
    hs : Membership.mem (Set.preimage Finset.Colex.ofColex ↑𝒜) (𝒜.sup' h𝒜₀ Finset. …
    t : Finset α
    ⊢ Iff (Membership.mem 𝒜 t) (Membership.mem (Finset.Colex.initSeg (𝒜.sup' h𝒜₀ F …
  -/
  rw [mem_initSeg]
  /-
    case h
    α : Type u_1
    inst✝¹ : LinearOrder α
    𝒜 : Finset (Finset α)
    r : Nat
    inst✝ : Fintype α
    h𝒜 : Finset.Colex.IsInitSeg 𝒜 r
    h𝒜₀ : 𝒜.Nonempty
    hs : Membership.mem (Set.preimage Finset.Colex.ofColex ↑𝒜) (𝒜.sup' h𝒜₀ Finset. …
    t : Finset α
    ⊢ Iff (Membership.mem 𝒜 t) (And (Eq (𝒜.sup' h𝒜₀ Finset.Colex.toColex).ofColex. …
  -/
  refine ⟨fun p ↦ ?_, ?_⟩
    /-
      case h.refine_1
      α : Type u_1
      inst✝¹ : LinearOrder α
      𝒜 : Finset (Finset α)
      r : Nat
      inst✝ : Fintype α
      h𝒜 : Finset.Colex.IsInitSeg 𝒜 r
      h𝒜₀ : 𝒜.Nonempty
      hs : Membership.mem (Set.preimage Finset.Colex.ofColex ↑𝒜) (𝒜.sup' h𝒜₀ Finset. …
      t : Finset α
      p : Membership.mem 𝒜 t
      ⊢ And (Eq (𝒜.sup' h𝒜₀ Finset.Colex.toColex).ofColex.card t.card) (LE.le { ofCo …
    -/
  · rw [h𝒜.1 p, h𝒜.1 hs]
    /-
      case h.refine_1
      α : Type u_1
      inst✝¹ : LinearOrder α
      𝒜 : Finset (Finset α)
      r : Nat
      inst✝ : Fintype α
      h𝒜 : Finset.Colex.IsInitSeg 𝒜 r
      h𝒜₀ : 𝒜.Nonempty
      hs : Membership.mem (Set.preimage Finset.Colex.ofColex ↑𝒜) (𝒜.sup' h𝒜₀ Finset. …
      t : Finset α
      p : Membership.mem 𝒜 t
      ⊢ And (Eq r r) (LE.le { ofColex := t } { ofColex := (𝒜.sup' h𝒜₀ Finset.Colex.t …
    -/
    exact ⟨rfl, le_sup' _ p⟩
    /-
      🎉 no goals
    -/
  /-
    case h.refine_2
    α : Type u_1
    inst✝¹ : LinearOrder α
    𝒜 : Finset (Finset α)
    r : Nat
    inst✝ : Fintype α
    h𝒜 : Finset.Colex.IsInitSeg 𝒜 r
    h𝒜₀ : 𝒜.Nonempty
    hs : Membership.mem (Set.preimage Finset.Colex.ofColex ↑𝒜) (𝒜.sup' h𝒜₀ Finset. …
    t : Finset α
    ⊢ And (Eq (𝒜.sup' h𝒜₀ Finset.Colex.toColex).ofColex.card t.card) (LE.le { ofCo …
  -/
  rintro ⟨cards, le⟩
  /-
    case h.refine_2.intro
    α : Type u_1
    inst✝¹ : LinearOrder α
    𝒜 : Finset (Finset α)
    r : Nat
    inst✝ : Fintype α
    h𝒜 : Finset.Colex.IsInitSeg 𝒜 r
    h𝒜₀ : 𝒜.Nonempty
    hs : Membership.mem (Set.preimage Finset.Colex.ofColex ↑𝒜) (𝒜.sup' h𝒜₀ Finset. …
    t : Finset α
    cards : Eq (𝒜.sup' h𝒜₀ Finset.Colex.toColex).ofColex.card t.card
    le : LE.le { ofColex := t } { ofColex := (𝒜.sup' h𝒜₀ Finset.Colex.toColex).ofC …
    ⊢ Membership.mem 𝒜 t
  -/
  obtain p | p := le.eq_or_lt
    /-
      case h.refine_2.intro.inl
      α : Type u_1
      inst✝¹ : LinearOrder α
      𝒜 : Finset (Finset α)
      r : Nat
      inst✝ : Fintype α
      h𝒜 : Finset.Colex.IsInitSeg 𝒜 r
      h𝒜₀ : 𝒜.Nonempty
      hs : Membership.mem (Set.preimage Finset.Colex.ofColex ↑𝒜) (𝒜.sup' h𝒜₀ Finset. …
      t : Finset α
      cards : Eq (𝒜.sup' h𝒜₀ Finset.Colex.toColex).ofColex.card t.card
      le : LE.le { ofColex := t } { ofColex := (𝒜.sup' h𝒜₀ Finset.Colex.toColex).ofC …
      p : Eq { ofColex := t } { ofColex := (𝒜.sup' h𝒜₀ Finset.Colex.toColex).ofColex }
      ⊢ Membership.mem 𝒜 t
    -/
  · rwa [toColex_inj.1 p]
    /-
      🎉 no goals
    -/
    /-
      case h.refine_2.intro.inr
      α : Type u_1
      inst✝¹ : LinearOrder α
      𝒜 : Finset (Finset α)
      r : Nat
      inst✝ : Fintype α
      h𝒜 : Finset.Colex.IsInitSeg 𝒜 r
      h𝒜₀ : 𝒜.Nonempty
      hs : Membership.mem (Set.preimage Finset.Colex.ofColex ↑𝒜) (𝒜.sup' h𝒜₀ Finset. …
      t : Finset α
      cards : Eq (𝒜.sup' h𝒜₀ Finset.Colex.toColex).ofColex.card t.card
      le : LE.le { ofColex := t } { ofColex := (𝒜.sup' h𝒜₀ Finset.Colex.toColex).ofC …
      p : LT.lt { ofColex := t } { ofColex := (𝒜.sup' h𝒜₀ Finset.Colex.toColex).ofCo …
      ⊢ Membership.mem 𝒜 t
    -/
  · exact h𝒜.2 hs ⟨p, cards ▸ h𝒜.1 hs⟩
    /-
      🎉 no goals
    -/


/-- Being a nonempty initial segment of colex is equivalent to being an `initSeg`. -/
lemma isInitSeg_iff_exists_initSeg :
    IsInitSeg 𝒜 r ∧ 𝒜.Nonempty ↔ ∃ s : Finset α, #s = r ∧ 𝒜 = initSeg s := by
  /-
    α : Type u_1
    inst✝¹ : LinearOrder α
    𝒜 : Finset (Finset α)
    r : Nat
    inst✝ : Fintype α
    ⊢ Iff (And (Finset.Colex.IsInitSeg 𝒜 r) 𝒜.Nonempty) (Exists fun s => And (Eq s …
  -/
  refine ⟨fun h𝒜 ↦ h𝒜.1.exists_initSeg h𝒜.2, ?_⟩
  /-
    α : Type u_1
    inst✝¹ : LinearOrder α
    𝒜 : Finset (Finset α)
    r : Nat
    inst✝ : Fintype α
    ⊢ (Exists fun s => And (Eq s.card r) (Eq 𝒜 (Finset.Colex.initSeg s))) → And (F …
  -/
  rintro ⟨s, rfl, rfl⟩
  /-
    case intro.intro
    α : Type u_1
    inst✝¹ : LinearOrder α
    inst✝ : Fintype α
    s : Finset α
    ⊢ And (Finset.Colex.IsInitSeg (Finset.Colex.initSeg s) s.card) (Finset.Colex.i …
  -/
  exact ⟨isInitSeg_initSeg, initSeg_nonempty⟩
  /-
    🎉 no goals
  -/


lemma geomSum_ofColex_strictMono (hn : 2 ≤ n) : StrictMono fun s ↦ ∑ k ∈ ofColex s, n ^ k := by
  /-
    n : Nat
    hn : LE.le 2 n
    ⊢ StrictMono fun s => s.ofColex.sum fun k => HPow.hPow n k
  -/
  rintro ⟨s⟩ ⟨t⟩ hst
  /-
    case toColex.toColex
    n : Nat
    hn : LE.le 2 n
    s t : Finset Nat
    hst : LT.lt { ofColex := s } { ofColex := t }
    ⊢ LT.lt ((fun s => s.ofColex.sum fun k => HPow.hPow n k) { ofColex := s }) ((f …
  -/
  rw [toColex_lt_toColex_iff_exists_forall_lt] at hst
  /-
    case toColex.toColex
    n : Nat
    hn : LE.le 2 n
    s t : Finset Nat
    hst : Exists fun a => And (Membership.mem t a) (And (Not (Membership.mem s a)) …
    ⊢ LT.lt ((fun s => s.ofColex.sum fun k => HPow.hPow n k) { ofColex := s }) ((f …
  -/
  obtain ⟨a, hat, has, ha⟩ := hst
  /-
    case toColex.toColex.intro.intro.intro
    n : Nat
    hn : LE.le 2 n
    s t : Finset Nat
    a : Nat
    hat : Membership.mem t a
    has : Not (Membership.mem s a)
    ha : ∀ (b : Nat), Membership.mem s b → Not (Membership.mem t b) → LT.lt b a
    ⊢ LT.lt ((fun s => s.ofColex.sum fun k => HPow.hPow n k) { ofColex := s }) ((f …
  -/
  rw [← sum_sdiff_lt_sum_sdiff]
  exact (Nat.geomSum_lt hn <| by simpa).trans_le <| single_le_sum (fun _ _ ↦ by positivity) <|
    mem_sdiff.2 ⟨hat, has⟩


/-- For finsets of naturals, the colexicographic order is equivalent to the order induced by the
`n`-ary expansion. -/
lemma geomSum_le_geomSum_iff_toColex_le_toColex (hn : 2 ≤ n) :
    ∑ k ∈ s, n ^ k ≤ ∑ k ∈ t, n ^ k ↔ toColex s ≤ toColex t :=
  (geomSum_ofColex_strictMono hn).le_iff_le


/-- For finsets of naturals, the colexicographic order is equivalent to the order induced by the
`n`-ary expansion. -/
lemma geomSum_lt_geomSum_iff_toColex_lt_toColex (hn : 2 ≤ n) :
    ∑ i ∈ s, n ^ i < ∑ i ∈ t, n ^ i ↔ toColex s < toColex t :=
  (geomSum_ofColex_strictMono hn).lt_iff_lt


theorem geomSum_injective {n : ℕ} (hn : 2 ≤ n) :
    Function.Injective (fun s : Finset ℕ ↦ ∑ i in s, n ^ i) := by
  /-
    n : Nat
    hn : LE.le 2 n
    ⊢ Function.Injective fun s => s.sum fun i => HPow.hPow n i
  -/
  intro _ _ h
  rwa [le_antisymm_iff, geomSum_le_geomSum_iff_toColex_le_toColex hn,
    geomSum_le_geomSum_iff_toColex_le_toColex hn, ← le_antisymm_iff, Colex.toColex.injEq] at h


theorem lt_geomSum_of_mem {a : ℕ} (hn : 2 ≤ n) (hi : a ∈ s) : a < ∑ i in s, n ^ i :=
                                                   /-
                                                     s : Finset Nat
                                                     n a : Nat
                                                     hn : LE.le 2 n
                                                     hi : Membership.mem s a
                                                     ⊢ ∀ (i : Nat), Membership.mem s i → LE.le 0 (HPow.hPow n i)
                                                   -/
  (a.lt_pow_self hn).trans_le <| single_le_sum (by simp) hi
                                                   /-
                                                     🎉 no goals
                                                   -/


@[simp] theorem toFinset_bitIndices_twoPowSum (s : Finset ℕ) :
    (∑ i in s, 2 ^ i).bitIndices.toFinset = s := by
  /-
    s : Finset Nat
    ⊢ Eq (s.sum fun i => HPow.hPow 2 i).bitIndices.toFinset s
  -/
  simp [← (geomSum_injective rfl.le).eq_iff, List.sum_toFinset _ Nat.bitIndices_sorted.nodup]
  /-
    🎉 no goals
  -/


@[simp] theorem twoPowSum_toFinset_bitIndices (n : ℕ) :
    ∑ i in n.bitIndices.toFinset, 2 ^ i = n := by
  /-
    n : Nat
    ⊢ Eq (n.bitIndices.toFinset.sum fun i => HPow.hPow 2 i) n
  -/
  simp [List.sum_toFinset _ Nat.bitIndices_sorted.nodup]
  /-
    🎉 no goals
  -/


/-- The equivalence between `ℕ` and `Finset ℕ` that maps `∑ i in s, 2^i` to `s`. -/
@[simps] def equivBitIndices : ℕ ≃ Finset ℕ where
  toFun n := n.bitIndices.toFinset
  invFun s := ∑ i in s, 2^i
  left_inv := twoPowSum_toFinset_bitIndices
  right_inv := toFinset_bitIndices_twoPowSum


/-- The equivalence `Nat.equivBitIndices` enumerates `Finset ℕ` in colexicographic order. -/
@[simps] def orderIsoColex : ℕ ≃o Colex ℕ where
  toFun n := Colex.toColex (equivBitIndices n)
  invFun s := equivBitIndices.symm s.ofColex
  left_inv n := equivBitIndices.symm_apply_apply n
  right_inv s :=  Finset.toColex_inj.2 (equivBitIndices.apply_symm_apply s.ofColex)
  map_rel_iff' := by simp [← (Finset.geomSum_le_geomSum_iff_toColex_le_toColex rfl.le),
    toFinset_bitIndices_twoPowSum]


