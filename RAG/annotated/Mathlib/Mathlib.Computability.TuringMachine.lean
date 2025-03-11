/-- The `BlankExtends` partial order holds of `l₁` and `l₂` if `l₂` is obtained by adding
blanks (`default : Γ`) to the end of `l₁`. -/
def BlankExtends {Γ} [Inhabited Γ] (l₁ l₂ : List Γ) : Prop :=
  ∃ n, l₂ = l₁ ++ List.replicate n default


@[refl]
theorem BlankExtends.refl {Γ} [Inhabited Γ] (l : List Γ) : BlankExtends l l :=
         /-
           Γ : Type u_1
           inst✝ : Inhabited Γ
           l : List Γ
           ⊢ Eq l (HAppend.hAppend l (List.replicate 0 Inhabited.default))
         -/
  ⟨0, by simp⟩
         /-
           🎉 no goals
         -/


@[trans]
theorem BlankExtends.trans {Γ} [Inhabited Γ] {l₁ l₂ l₃ : List Γ} :
    BlankExtends l₁ l₂ → BlankExtends l₂ l₃ → BlankExtends l₁ l₃ := by
  /-
    Γ : Type u_1
    inst✝ : Inhabited Γ
    l₁ l₂ l₃ : List Γ
    ⊢ Turing.BlankExtends l₁ l₂ → Turing.BlankExtends l₂ l₃ → Turing.BlankExtends  …
  -/
  rintro ⟨i, rfl⟩ ⟨j, rfl⟩
  /-
    case intro.intro
    Γ : Type u_1
    inst✝ : Inhabited Γ
    l₁ : List Γ
    i j : Nat
    ⊢ Turing.BlankExtends l₁ (HAppend.hAppend (HAppend.hAppend l₁ (List.replicate  …
  -/
  exact ⟨i + j, by simp⟩
  /-
    🎉 no goals
  -/


theorem BlankExtends.below_of_le {Γ} [Inhabited Γ] {l l₁ l₂ : List Γ} :
    BlankExtends l l₁ → BlankExtends l l₂ → l₁.length ≤ l₂.length → BlankExtends l₁ l₂ := by
  /-
    Γ : Type u_1
    inst✝ : Inhabited Γ
    l l₁ l₂ : List Γ
    ⊢ Turing.BlankExtends l l₁ → Turing.BlankExtends l l₂ → LE.le l₁.length l₂.len …
  -/
  rintro ⟨i, rfl⟩ ⟨j, rfl⟩ h; use j - i
  /-
    case h
    Γ : Type u_1
    inst✝ : Inhabited Γ
    l : List Γ
    i j : Nat
    h : LE.le (HAppend.hAppend l (List.replicate i Inhabited.default)).length (HAp …
    ⊢ Eq (HAppend.hAppend l (List.replicate j Inhabited.default)) (HAppend.hAppend …
  -/
  simp only [List.length_append, Nat.add_le_add_iff_left, List.length_replicate] at h
  /-
    case h
    Γ : Type u_1
    inst✝ : Inhabited Γ
    l : List Γ
    i j : Nat
    h : LE.le i j
    ⊢ Eq (HAppend.hAppend l (List.replicate j Inhabited.default)) (HAppend.hAppend …
  -/
  simp only [← List.replicate_add, Nat.add_sub_cancel' h, List.append_assoc]
  /-
    🎉 no goals
  -/


/-- Any two extensions by blank `l₁,l₂` of `l` have a common join (which can be taken to be the
longer of `l₁` and `l₂`). -/
def BlankExtends.above {Γ} [Inhabited Γ] {l l₁ l₂ : List Γ} (h₁ : BlankExtends l l₁)
    (h₂ : BlankExtends l l₂) : { l' // BlankExtends l₁ l' ∧ BlankExtends l₂ l' } :=
  if h : l₁.length ≤ l₂.length then ⟨l₂, h₁.below_of_le h₂ h, BlankExtends.refl _⟩
  else ⟨l₁, BlankExtends.refl _, h₂.below_of_le h₁ (le_of_not_ge h)⟩


theorem BlankExtends.above_of_le {Γ} [Inhabited Γ] {l l₁ l₂ : List Γ} :
    BlankExtends l₁ l → BlankExtends l₂ l → l₁.length ≤ l₂.length → BlankExtends l₁ l₂ := by
  /-
    Γ : Type u_1
    inst✝ : Inhabited Γ
    l l₁ l₂ : List Γ
    ⊢ Turing.BlankExtends l₁ l → Turing.BlankExtends l₂ l → LE.le l₁.length l₂.len …
  -/
  rintro ⟨i, rfl⟩ ⟨j, e⟩ h; use i - j
  /-
    case h
    Γ : Type u_1
    inst✝ : Inhabited Γ
    l₁ l₂ : List Γ
    i j : Nat
    e : Eq (HAppend.hAppend l₁ (List.replicate i Inhabited.default)) (HAppend.hApp …
    h : LE.le l₁.length l₂.length
    ⊢ Eq l₂ (HAppend.hAppend l₁ (List.replicate (HSub.hSub i j) Inhabited.default))
  -/
  refine List.append_cancel_right (e.symm.trans ?_)
  /-
    case h
    Γ : Type u_1
    inst✝ : Inhabited Γ
    l₁ l₂ : List Γ
    i j : Nat
    e : Eq (HAppend.hAppend l₁ (List.replicate i Inhabited.default)) (HAppend.hApp …
    h : LE.le l₁.length l₂.length
    ⊢ Eq (HAppend.hAppend l₁ (List.replicate i Inhabited.default)) (HAppend.hAppen …
  -/
  rw [List.append_assoc, ← List.replicate_add, Nat.sub_add_cancel]
  /-
    case h
    Γ : Type u_1
    inst✝ : Inhabited Γ
    l₁ l₂ : List Γ
    i j : Nat
    e : Eq (HAppend.hAppend l₁ (List.replicate i Inhabited.default)) (HAppend.hApp …
    h : LE.le l₁.length l₂.length
    ⊢ LE.le j i
  -/
  apply_fun List.length at e
  /-
    case h
    Γ : Type u_1
    inst✝ : Inhabited Γ
    l₁ l₂ : List Γ
    i j : Nat
    h : LE.le l₁.length l₂.length
    e : Eq (HAppend.hAppend l₁ (List.replicate i Inhabited.default)).length (HAppe …
    ⊢ LE.le j i
  -/
  simp only [List.length_append, List.length_replicate] at e
  /-
    case h
    Γ : Type u_1
    inst✝ : Inhabited Γ
    l₁ l₂ : List Γ
    i j : Nat
    h : LE.le l₁.length l₂.length
    e : Eq (HAdd.hAdd l₁.length i) (HAdd.hAdd l₂.length j)
    ⊢ LE.le j i
  -/
  rwa [← Nat.add_le_add_iff_left, e, Nat.add_le_add_iff_right]
  /-
    🎉 no goals
  -/


/-- `BlankRel` is the symmetric closure of `BlankExtends`, turning it into an equivalence
relation. Two lists are related by `BlankRel` if one extends the other by blanks. -/
def BlankRel {Γ} [Inhabited Γ] (l₁ l₂ : List Γ) : Prop :=
  BlankExtends l₁ l₂ ∨ BlankExtends l₂ l₁


@[refl]
theorem BlankRel.refl {Γ} [Inhabited Γ] (l : List Γ) : BlankRel l l :=
  Or.inl (BlankExtends.refl _)


@[symm]
theorem BlankRel.symm {Γ} [Inhabited Γ] {l₁ l₂ : List Γ} : BlankRel l₁ l₂ → BlankRel l₂ l₁ :=
  Or.symm


@[trans]
theorem BlankRel.trans {Γ} [Inhabited Γ] {l₁ l₂ l₃ : List Γ} :
    BlankRel l₁ l₂ → BlankRel l₂ l₃ → BlankRel l₁ l₃ := by
  /-
    Γ : Type u_1
    inst✝ : Inhabited Γ
    l₁ l₂ l₃ : List Γ
    ⊢ Turing.BlankRel l₁ l₂ → Turing.BlankRel l₂ l₃ → Turing.BlankRel l₁ l₃
  -/
  rintro (h₁ | h₁) (h₂ | h₂)
    /-
      case inl.inl
      Γ : Type u_1
      inst✝ : Inhabited Γ
      l₁ l₂ l₃ : List Γ
      h₁ : Turing.BlankExtends l₁ l₂
      h₂ : Turing.BlankExtends l₂ l₃
      ⊢ Turing.BlankRel l₁ l₃
    -/
  · exact Or.inl (h₁.trans h₂)
    /-
      🎉 no goals
    -/
    /-
      case inl.inr
      Γ : Type u_1
      inst✝ : Inhabited Γ
      l₁ l₂ l₃ : List Γ
      h₁ : Turing.BlankExtends l₁ l₂
      h₂ : Turing.BlankExtends l₃ l₂
      ⊢ Turing.BlankRel l₁ l₃
    -/
  · rcases le_total l₁.length l₃.length with h | h
      /-
        case inl.inr.inl
        Γ : Type u_1
        inst✝ : Inhabited Γ
        l₁ l₂ l₃ : List Γ
        h₁ : Turing.BlankExtends l₁ l₂
        h₂ : Turing.BlankExtends l₃ l₂
        h : LE.le l₁.length l₃.length
        ⊢ Turing.BlankRel l₁ l₃
      -/
    · exact Or.inl (h₁.above_of_le h₂ h)
      /-
        🎉 no goals
      -/
      /-
        case inl.inr.inr
        Γ : Type u_1
        inst✝ : Inhabited Γ
        l₁ l₂ l₃ : List Γ
        h₁ : Turing.BlankExtends l₁ l₂
        h₂ : Turing.BlankExtends l₃ l₂
        h : LE.le l₃.length l₁.length
        ⊢ Turing.BlankRel l₁ l₃
      -/
    · exact Or.inr (h₂.above_of_le h₁ h)
      /-
        🎉 no goals
      -/
    /-
      case inr.inl
      Γ : Type u_1
      inst✝ : Inhabited Γ
      l₁ l₂ l₃ : List Γ
      h₁ : Turing.BlankExtends l₂ l₁
      h₂ : Turing.BlankExtends l₂ l₃
      ⊢ Turing.BlankRel l₁ l₃
    -/
  · rcases le_total l₁.length l₃.length with h | h
      /-
        case inr.inl.inl
        Γ : Type u_1
        inst✝ : Inhabited Γ
        l₁ l₂ l₃ : List Γ
        h₁ : Turing.BlankExtends l₂ l₁
        h₂ : Turing.BlankExtends l₂ l₃
        h : LE.le l₁.length l₃.length
        ⊢ Turing.BlankRel l₁ l₃
      -/
    · exact Or.inl (h₁.below_of_le h₂ h)
      /-
        🎉 no goals
      -/
      /-
        case inr.inl.inr
        Γ : Type u_1
        inst✝ : Inhabited Γ
        l₁ l₂ l₃ : List Γ
        h₁ : Turing.BlankExtends l₂ l₁
        h₂ : Turing.BlankExtends l₂ l₃
        h : LE.le l₃.length l₁.length
        ⊢ Turing.BlankRel l₁ l₃
      -/
    · exact Or.inr (h₂.below_of_le h₁ h)
      /-
        🎉 no goals
      -/
    /-
      case inr.inr
      Γ : Type u_1
      inst✝ : Inhabited Γ
      l₁ l₂ l₃ : List Γ
      h₁ : Turing.BlankExtends l₂ l₁
      h₂ : Turing.BlankExtends l₃ l₂
      ⊢ Turing.BlankRel l₁ l₃
    -/
  · exact Or.inr (h₂.trans h₁)
    /-
      🎉 no goals
    -/


/-- Given two `BlankRel` lists, there exists (constructively) a common join. -/
def BlankRel.above {Γ} [Inhabited Γ] {l₁ l₂ : List Γ} (h : BlankRel l₁ l₂) :
    { l // BlankExtends l₁ l ∧ BlankExtends l₂ l } := by
  refine
    if hl : l₁.length ≤ l₂.length then ⟨l₂, Or.elim h id fun h' ↦ ?_, BlankExtends.refl _⟩
    else ⟨l₁, BlankExtends.refl _, Or.elim h (fun h' ↦ ?_) id⟩
    /-
      case refine_1
      Γ : Type ?u.7015
      inst✝ : Inhabited Γ
      l₁ l₂ : List Γ
      h : Turing.BlankRel l₁ l₂
      hl : LE.le l₁.length l₂.length
      h' : Turing.BlankExtends l₂ l₁
      ⊢ Turing.BlankExtends l₁ l₂
    -/
  · exact (BlankExtends.refl _).above_of_le h' hl
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      Γ : Type ?u.7015
      inst✝ : Inhabited Γ
      l₁ l₂ : List Γ
      h : Turing.BlankRel l₁ l₂
      hl : Not (LE.le l₁.length l₂.length)
      h' : Turing.BlankExtends l₁ l₂
      ⊢ Turing.BlankExtends l₂ l₁
    -/
  · exact (BlankExtends.refl _).above_of_le h' (le_of_not_ge hl)
    /-
      🎉 no goals
    -/


/-- Given two `BlankRel` lists, there exists (constructively) a common meet. -/
def BlankRel.below {Γ} [Inhabited Γ] {l₁ l₂ : List Γ} (h : BlankRel l₁ l₂) :
    { l // BlankExtends l l₁ ∧ BlankExtends l l₂ } := by
  refine
    if hl : l₁.length ≤ l₂.length then ⟨l₁, BlankExtends.refl _, Or.elim h id fun h' ↦ ?_⟩
    else ⟨l₂, Or.elim h (fun h' ↦ ?_) id, BlankExtends.refl _⟩
    /-
      case refine_1
      Γ : Type ?u.8431
      inst✝ : Inhabited Γ
      l₁ l₂ : List Γ
      h : Turing.BlankRel l₁ l₂
      hl : LE.le l₁.length l₂.length
      h' : Turing.BlankExtends l₂ l₁
      ⊢ Turing.BlankExtends l₁ l₂
    -/
  · exact (BlankExtends.refl _).above_of_le h' hl
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      Γ : Type ?u.8431
      inst✝ : Inhabited Γ
      l₁ l₂ : List Γ
      h : Turing.BlankRel l₁ l₂
      hl : Not (LE.le l₁.length l₂.length)
      h' : Turing.BlankExtends l₁ l₂
      ⊢ Turing.BlankExtends l₂ l₁
    -/
  · exact (BlankExtends.refl _).above_of_le h' (le_of_not_ge hl)
    /-
      🎉 no goals
    -/


theorem BlankRel.equivalence (Γ) [Inhabited Γ] : Equivalence (@BlankRel Γ _) :=
  ⟨BlankRel.refl, @BlankRel.symm _ _, @BlankRel.trans _ _⟩


/-- Construct a setoid instance for `BlankRel`. -/
def BlankRel.setoid (Γ) [Inhabited Γ] : Setoid (List Γ) :=
  ⟨_, BlankRel.equivalence _⟩


/-- A `ListBlank Γ` is a quotient of `List Γ` by extension by blanks at the end. This is used to
represent half-tapes of a Turing machine, so that we can pretend that the list continues
infinitely with blanks. -/
def ListBlank (Γ) [Inhabited Γ] :=
  Quotient (BlankRel.setoid Γ)


instance ListBlank.inhabited {Γ} [Inhabited Γ] : Inhabited (ListBlank Γ) :=
  ⟨Quotient.mk'' []⟩


instance ListBlank.hasEmptyc {Γ} [Inhabited Γ] : EmptyCollection (ListBlank Γ) :=
  ⟨Quotient.mk'' []⟩


/-- A modified version of `Quotient.liftOn'` specialized for `ListBlank`, with the stronger
precondition `BlankExtends` instead of `BlankRel`. -/
protected abbrev ListBlank.liftOn {Γ} [Inhabited Γ] {α} (l : ListBlank Γ) (f : List Γ → α)
    (H : ∀ a b, BlankExtends a b → f a = f b) : α :=
                    /-
                      Γ : Type ?u.10752
                      inst✝ : Inhabited Γ
                      α : Sort ?u.10768
                      l : Turing.ListBlank Γ
                      f : List Γ → α
                      H : ∀ (a b : List Γ), Turing.BlankExtends a b → Eq (f a) (f b)
                      ⊢ ∀ (a b : List Γ), (Turing.BlankRel.setoid Γ) a b → Eq (f a) (f b)
                    -/
  l.liftOn' f <| by rintro a b (h | h) <;> [exact H _ _ h; exact (H _ _ h).symm]
                    /-
                      🎉 no goals
                    -/


/-- The quotient map turning a `List` into a `ListBlank`. -/
def ListBlank.mk {Γ} [Inhabited Γ] : List Γ → ListBlank Γ :=
  Quotient.mk''


@[elab_as_elim]
protected theorem ListBlank.induction_on {Γ} [Inhabited Γ] {p : ListBlank Γ → Prop}
    (q : ListBlank Γ) (h : ∀ a, p (ListBlank.mk a)) : p q :=
  Quotient.inductionOn' q h


/-- The head of a `ListBlank` is well defined. -/
def ListBlank.head {Γ} [Inhabited Γ] (l : ListBlank Γ) : Γ := by
  /-
    Γ : Type ?u.11709
    inst✝ : Inhabited Γ
    l : Turing.ListBlank Γ
    ⊢ Γ
  -/
  apply l.liftOn List.headI
  /-
    Γ : Type ?u.11709
    inst✝ : Inhabited Γ
    l : Turing.ListBlank Γ
    ⊢ ∀ (a b : List Γ), Turing.BlankExtends a b → Eq a.headI b.headI
  -/
  rintro a _ ⟨i, rfl⟩
  /-
    case intro
    Γ : Type ?u.11709
    inst✝ : Inhabited Γ
    l : Turing.ListBlank Γ
    a : List Γ
    i : Nat
    ⊢ Eq a.headI (HAppend.hAppend a (List.replicate i Inhabited.default)).headI
  -/
  cases a
    /-
      case intro.nil
      Γ : Type ?u.11709
      inst✝ : Inhabited Γ
      l : Turing.ListBlank Γ
      i : Nat
      ⊢ Eq List.nil.headI (HAppend.hAppend List.nil (List.replicate i Inhabited.defa …
    -/
                /-
                  🎉 no goals
                -/
  · cases i <;> rfl
                /-
                  🎉 no goals
                -/
  /-
    case intro.cons
    Γ : Type ?u.11709
    inst✝ : Inhabited Γ
    l : Turing.ListBlank Γ
    i : Nat
    head✝ : Γ
    tail✝ : List Γ
    ⊢ Eq (List.cons head✝ tail✝).headI (HAppend.hAppend (List.cons head✝ tail✝) (L …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem ListBlank.head_mk {Γ} [Inhabited Γ] (l : List Γ) :
    ListBlank.head (ListBlank.mk l) = l.headI :=
  rfl


/-- The tail of a `ListBlank` is well defined (up to the tail of blanks). -/
def ListBlank.tail {Γ} [Inhabited Γ] (l : ListBlank Γ) : ListBlank Γ := by
  /-
    Γ : Type ?u.12453
    inst✝ : Inhabited Γ
    l : Turing.ListBlank Γ
    ⊢ Turing.ListBlank Γ
  -/
  apply l.liftOn (fun l ↦ ListBlank.mk l.tail)
  /-
    Γ : Type ?u.12453
    inst✝ : Inhabited Γ
    l : Turing.ListBlank Γ
    ⊢ ∀ (a b : List Γ), Turing.BlankExtends a b → Eq (Turing.ListBlank.mk a.tail)  …
  -/
  rintro a _ ⟨i, rfl⟩
  /-
    case intro
    Γ : Type ?u.12453
    inst✝ : Inhabited Γ
    l : Turing.ListBlank Γ
    a : List Γ
    i : Nat
    ⊢ Eq (Turing.ListBlank.mk a.tail) (Turing.ListBlank.mk (HAppend.hAppend a (Lis …
  -/
  refine Quotient.sound' (Or.inl ?_)
  /-
    case intro
    Γ : Type ?u.12453
    inst✝ : Inhabited Γ
    l : Turing.ListBlank Γ
    a : List Γ
    i : Nat
    ⊢ Turing.BlankExtends a.tail (HAppend.hAppend a (List.replicate i Inhabited.de …
  -/
  cases a
    /-
      case intro.nil
      Γ : Type ?u.12453
      inst✝ : Inhabited Γ
      l : Turing.ListBlank Γ
      i : Nat
      ⊢ Turing.BlankExtends List.nil.tail (HAppend.hAppend List.nil (List.replicate  …
    -/
  · cases' i with i <;> [exact ⟨0, rfl⟩; exact ⟨i, rfl⟩]
    /-
      🎉 no goals
    -/
  /-
    case intro.cons
    Γ : Type ?u.12453
    inst✝ : Inhabited Γ
    l : Turing.ListBlank Γ
    i : Nat
    head✝ : Γ
    tail✝ : List Γ
    ⊢ Turing.BlankExtends (List.cons head✝ tail✝).tail (HAppend.hAppend (List.cons …
  -/
  exact ⟨i, rfl⟩
  /-
    🎉 no goals
  -/


@[simp]
theorem ListBlank.tail_mk {Γ} [Inhabited Γ] (l : List Γ) :
    ListBlank.tail (ListBlank.mk l) = ListBlank.mk l.tail :=
  rfl


/-- We can cons an element onto a `ListBlank`. -/
def ListBlank.cons {Γ} [Inhabited Γ] (a : Γ) (l : ListBlank Γ) : ListBlank Γ := by
  /-
    Γ : Type ?u.13465
    inst✝ : Inhabited Γ
    a : Γ
    l : Turing.ListBlank Γ
    ⊢ Turing.ListBlank Γ
  -/
  apply l.liftOn (fun l ↦ ListBlank.mk (List.cons a l))
  /-
    Γ : Type ?u.13465
    inst✝ : Inhabited Γ
    a : Γ
    l : Turing.ListBlank Γ
    ⊢ ∀ (a_1 b : List Γ), Turing.BlankExtends a_1 b → Eq (Turing.ListBlank.mk (Lis …
  -/
  rintro _ _ ⟨i, rfl⟩
  /-
    case intro
    Γ : Type ?u.13465
    inst✝ : Inhabited Γ
    a : Γ
    l : Turing.ListBlank Γ
    a✝ : List Γ
    i : Nat
    ⊢ Eq (Turing.ListBlank.mk (List.cons a a✝)) (Turing.ListBlank.mk (List.cons a  …
  -/
  exact Quotient.sound' (Or.inl ⟨i, rfl⟩)
  /-
    🎉 no goals
  -/


@[simp]
theorem ListBlank.cons_mk {Γ} [Inhabited Γ] (a : Γ) (l : List Γ) :
    ListBlank.cons a (ListBlank.mk l) = ListBlank.mk (a :: l) :=
  rfl


@[simp]
theorem ListBlank.head_cons {Γ} [Inhabited Γ] (a : Γ) : ∀ l : ListBlank Γ, (l.cons a).head = a :=
  Quotient.ind' fun _ ↦ rfl


@[simp]
theorem ListBlank.tail_cons {Γ} [Inhabited Γ] (a : Γ) : ∀ l : ListBlank Γ, (l.cons a).tail = l :=
  Quotient.ind' fun _ ↦ rfl


/-- The `cons` and `head`/`tail` functions are mutually inverse, unlike in the case of `List` where
this only holds for nonempty lists. -/
@[simp]
theorem ListBlank.cons_head_tail {Γ} [Inhabited Γ] : ∀ l : ListBlank Γ, l.tail.cons l.head = l := by
  /-
    Γ : Type u_1
    inst✝ : Inhabited Γ
    ⊢ ∀ (l : Turing.ListBlank Γ), Eq (Turing.ListBlank.cons l.head l.tail) l
  -/
  apply Quotient.ind'
  /-
    case h
    Γ : Type u_1
    inst✝ : Inhabited Γ
    ⊢ ∀ (a : List Γ), Eq (Turing.ListBlank.cons (Turing.ListBlank.head (Quotient.m …
  -/
  refine fun l ↦ Quotient.sound' (Or.inr ?_)
  /-
    case h
    Γ : Type u_1
    inst✝ : Inhabited Γ
    l : List Γ
    ⊢ Turing.BlankExtends l (List.cons (Turing.ListBlank.head (Quotient.mk'' l)) l …
  -/
  cases l
    /-
      case h.nil
      Γ : Type u_1
      inst✝ : Inhabited Γ
      ⊢ Turing.BlankExtends List.nil (List.cons (Turing.ListBlank.head (Quotient.mk' …
    -/
  · exact ⟨1, rfl⟩
    /-
      🎉 no goals
    -/
    /-
      case h.cons
      Γ : Type u_1
      inst✝ : Inhabited Γ
      head✝ : Γ
      tail✝ : List Γ
      ⊢ Turing.BlankExtends (List.cons head✝ tail✝) (List.cons (Turing.ListBlank.hea …
    -/
  · rfl
    /-
      🎉 no goals
    -/


/-- The `cons` and `head`/`tail` functions are mutually inverse, unlike in the case of `List` where
this only holds for nonempty lists. -/
theorem ListBlank.exists_cons {Γ} [Inhabited Γ] (l : ListBlank Γ) :
    ∃ a l', l = ListBlank.cons a l' :=
  ⟨_, _, (ListBlank.cons_head_tail _).symm⟩


/-- The n-th element of a `ListBlank` is well defined for all `n : ℕ`, unlike in a `List`. -/
def ListBlank.nth {Γ} [Inhabited Γ] (l : ListBlank Γ) (n : ℕ) : Γ := by
  /-
    Γ : Type ?u.15165
    inst✝ : Inhabited Γ
    l : Turing.ListBlank Γ
    n : Nat
    ⊢ Γ
  -/
  apply l.liftOn (fun l ↦ List.getI l n)
  /-
    Γ : Type ?u.15165
    inst✝ : Inhabited Γ
    l : Turing.ListBlank Γ
    n : Nat
    ⊢ ∀ (a b : List Γ), Turing.BlankExtends a b → Eq (a.getI n) (b.getI n)
  -/
  rintro l _ ⟨i, rfl⟩
  /-
    case intro
    Γ : Type ?u.15165
    inst✝ : Inhabited Γ
    l✝ : Turing.ListBlank Γ
    n : Nat
    l : List Γ
    i : Nat
    ⊢ Eq (l.getI n) ((HAppend.hAppend l (List.replicate i Inhabited.default)).getI …
  -/
  cases' lt_or_le n _ with h h
    /-
      case intro.inl
      Γ : Type ?u.15165
      inst✝ : Inhabited Γ
      l✝ : Turing.ListBlank Γ
      n : Nat
      l : List Γ
      i : Nat
      h : LT.lt n ?m.15439
      ⊢ Eq (l.getI n) ((HAppend.hAppend l (List.replicate i Inhabited.default)).getI …
    -/
  · rw [List.getI_append _ _ _ h]
    /-
      🎉 no goals
    -/
  /-
    case intro.inr
    Γ : Type ?u.15165
    inst✝ : Inhabited Γ
    l✝ : Turing.ListBlank Γ
    n : Nat
    l : List Γ
    i : Nat
    h : LE.le l.length n
    ⊢ Eq (l.getI n) ((HAppend.hAppend l (List.replicate i Inhabited.default)).getI …
  -/
  rw [List.getI_eq_default _ h]
  /-
    case intro.inr
    Γ : Type ?u.15165
    inst✝ : Inhabited Γ
    l✝ : Turing.ListBlank Γ
    n : Nat
    l : List Γ
    i : Nat
    h : LE.le l.length n
    ⊢ Eq Inhabited.default ((HAppend.hAppend l (List.replicate i Inhabited.default …
  -/
  rcases le_or_lt _ n with h₂ | h₂
    /-
      case intro.inr.inl
      Γ : Type ?u.15165
      inst✝ : Inhabited Γ
      l✝ : Turing.ListBlank Γ
      n : Nat
      l : List Γ
      i : Nat
      h : LE.le l.length n
      h₂ : LE.le ?m.15889 n
      ⊢ Eq Inhabited.default ((HAppend.hAppend l (List.replicate i Inhabited.default …
    -/
  · rw [List.getI_eq_default _ h₂]
    /-
      🎉 no goals
    -/
  /-
    case intro.inr.inr
    Γ : Type ?u.15165
    inst✝ : Inhabited Γ
    l✝ : Turing.ListBlank Γ
    n : Nat
    l : List Γ
    i : Nat
    h : LE.le l.length n
    h₂ : LT.lt n (HAppend.hAppend l (List.replicate i Inhabited.default)).length
    ⊢ Eq Inhabited.default ((HAppend.hAppend l (List.replicate i Inhabited.default …
  -/
  rw [List.getI_eq_getElem _ h₂, List.getElem_append_right h, List.getElem_replicate]
  /-
    🎉 no goals
  -/


@[simp]
theorem ListBlank.nth_mk {Γ} [Inhabited Γ] (l : List Γ) (n : ℕ) :
    (ListBlank.mk l).nth n = l.getI n :=
  rfl


@[simp]
theorem ListBlank.nth_zero {Γ} [Inhabited Γ] (l : ListBlank Γ) : l.nth 0 = l.head := by
  /-
    Γ : Type u_1
    inst✝ : Inhabited Γ
    l : Turing.ListBlank Γ
    ⊢ Eq (l.nth 0) l.head
  -/
  conv => lhs; rw [← ListBlank.cons_head_tail l]
  /-
    Γ : Type u_1
    inst✝ : Inhabited Γ
    l : Turing.ListBlank Γ
    ⊢ Eq ((Turing.ListBlank.cons l.head l.tail).nth 0) l.head
  -/
  exact Quotient.inductionOn' l.tail fun l ↦ rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem ListBlank.nth_succ {Γ} [Inhabited Γ] (l : ListBlank Γ) (n : ℕ) :
    l.nth (n + 1) = l.tail.nth n := by
  /-
    Γ : Type u_1
    inst✝ : Inhabited Γ
    l : Turing.ListBlank Γ
    n : Nat
    ⊢ Eq (l.nth (HAdd.hAdd n 1)) (l.tail.nth n)
  -/
  conv => lhs; rw [← ListBlank.cons_head_tail l]
  /-
    Γ : Type u_1
    inst✝ : Inhabited Γ
    l : Turing.ListBlank Γ
    n : Nat
    ⊢ Eq ((Turing.ListBlank.cons l.head l.tail).nth (HAdd.hAdd n 1)) (l.tail.nth n)
  -/
  exact Quotient.inductionOn' l.tail fun l ↦ rfl
  /-
    🎉 no goals
  -/


@[ext]
theorem ListBlank.ext {Γ} [i : Inhabited Γ] {L₁ L₂ : ListBlank Γ} :
    (∀ i, L₁.nth i = L₂.nth i) → L₁ = L₂ := by
  /-
    Γ : Type u_1
    i : Inhabited Γ
    L₁ L₂ : Turing.ListBlank Γ
    ⊢ (∀ (i_1 : Nat), Eq (L₁.nth i_1) (L₂.nth i_1)) → Eq L₁ L₂
  -/
  refine ListBlank.induction_on L₁ fun l₁ ↦ ListBlank.induction_on L₂ fun l₂ H ↦ ?_
  /-
    Γ : Type u_1
    i : Inhabited Γ
    L₁ L₂ : Turing.ListBlank Γ
    l₁ l₂ : List Γ
    H : ∀ (i_1 : Nat), Eq ((Turing.ListBlank.mk l₁).nth i_1) ((Turing.ListBlank.mk …
    ⊢ Eq (Turing.ListBlank.mk l₁) (Turing.ListBlank.mk l₂)
  -/
  wlog h : l₁.length ≤ l₂.length
    /-
      case inr
      Γ : Type u_1
      i : Inhabited Γ
      L₁ L₂ : Turing.ListBlank Γ
      l₁ l₂ : List Γ
      H : ∀ (i_1 : Nat), Eq ((Turing.ListBlank.mk l₁).nth i_1) ((Turing.ListBlank.mk …
      this : ∀ {Γ : Type u_1} [i : Inhabited Γ] {L₁ L₂ : Turing.ListBlank Γ} (l₁ l₂  …
      h : Not (LE.le l₁.length l₂.length)
      ⊢ Eq (Turing.ListBlank.mk l₁) (Turing.ListBlank.mk l₂)
    -/
                                                                           /-
                                                                             🎉 no goals
                                                                           -/
                                                                           /-
                                                                             🎉 no goals
                                                                           -/
                                                                           /-
                                                                             🎉 no goals
                                                                           -/
                                                                           /-
                                                                             🎉 no goals
                                                                           -/
                                                                           /-
                                                                             🎉 no goals
                                                                           -/
                                                                           /-
                                                                             🎉 no goals
                                                                           -/
  · cases le_total l₁.length l₂.length <;> [skip; symm] <;> apply this <;> try assumption
                                                                           /-
                                                                             🎉 no goals
                                                                           -/
    /-
      case inr.inr.H
      Γ : Type u_1
      i : Inhabited Γ
      L₁ L₂ : Turing.ListBlank Γ
      l₁ l₂ : List Γ
      H : ∀ (i_1 : Nat), Eq ((Turing.ListBlank.mk l₁).nth i_1) ((Turing.ListBlank.mk …
      this : ∀ {Γ : Type u_1} [i : Inhabited Γ] {L₁ L₂ : Turing.ListBlank Γ} (l₁ l₂  …
      h : Not (LE.le l₁.length l₂.length)
      h✝ : LE.le l₂.length l₁.length
      ⊢ ∀ (i_1 : Nat), Eq ((Turing.ListBlank.mk l₂).nth i_1) ((Turing.ListBlank.mk l …
    -/
    intro
    /-
      case inr.inr.H
      Γ : Type u_1
      i : Inhabited Γ
      L₁ L₂ : Turing.ListBlank Γ
      l₁ l₂ : List Γ
      H : ∀ (i_1 : Nat), Eq ((Turing.ListBlank.mk l₁).nth i_1) ((Turing.ListBlank.mk …
      this : ∀ {Γ : Type u_1} [i : Inhabited Γ] {L₁ L₂ : Turing.ListBlank Γ} (l₁ l₂  …
      h : Not (LE.le l₁.length l₂.length)
      h✝ : LE.le l₂.length l₁.length
      i✝ : Nat
      ⊢ Eq ((Turing.ListBlank.mk l₂).nth i✝) ((Turing.ListBlank.mk l₁).nth i✝)
    -/
    rw [H]
    /-
      🎉 no goals
    -/
  /-
    Γ : Type u_1
    i : Inhabited Γ
    L₁ L₂ : Turing.ListBlank Γ
    l₁ l₂ : List Γ
    H : ∀ (i_1 : Nat), Eq ((Turing.ListBlank.mk l₁).nth i_1) ((Turing.ListBlank.mk …
    h : LE.le l₁.length l₂.length
    ⊢ Eq (Turing.ListBlank.mk l₁) (Turing.ListBlank.mk l₂)
  -/
  refine Quotient.sound' (Or.inl ⟨l₂.length - l₁.length, ?_⟩)
  /-
    Γ : Type u_1
    i : Inhabited Γ
    L₁ L₂ : Turing.ListBlank Γ
    l₁ l₂ : List Γ
    H : ∀ (i_1 : Nat), Eq ((Turing.ListBlank.mk l₁).nth i_1) ((Turing.ListBlank.mk …
    h : LE.le l₁.length l₂.length
    ⊢ Eq l₂ (HAppend.hAppend l₁ (List.replicate (HSub.hSub l₂.length l₁.length) In …
  -/
  refine List.ext_getElem ?_ fun i h h₂ ↦ Eq.symm ?_
    /-
      case refine_1
      Γ : Type u_1
      i : Inhabited Γ
      L₁ L₂ : Turing.ListBlank Γ
      l₁ l₂ : List Γ
      H : ∀ (i_1 : Nat), Eq ((Turing.ListBlank.mk l₁).nth i_1) ((Turing.ListBlank.mk …
      h : LE.le l₁.length l₂.length
      ⊢ Eq l₂.length (HAppend.hAppend l₁ (List.replicate (HSub.hSub l₂.length l₁.len …
    -/
  · simp only [Nat.add_sub_cancel' h, List.length_append, List.length_replicate]
    /-
      🎉 no goals
    -/
  /-
    case refine_2
    Γ : Type u_1
    i✝ : Inhabited Γ
    L₁ L₂ : Turing.ListBlank Γ
    l₁ l₂ : List Γ
    H : ∀ (i : Nat), Eq ((Turing.ListBlank.mk l₁).nth i) ((Turing.ListBlank.mk l₂) …
    h✝ : LE.le l₁.length l₂.length
    i : Nat
    h : LT.lt i l₂.length
    h₂ : LT.lt i (HAppend.hAppend l₁ (List.replicate (HSub.hSub l₂.length l₁.lengt …
    ⊢ Eq (GetElem.getElem (HAppend.hAppend l₁ (List.replicate (HSub.hSub l₂.length …
  -/
  simp only [ListBlank.nth_mk] at H
  /-
    case refine_2
    Γ : Type u_1
    i✝ : Inhabited Γ
    L₁ L₂ : Turing.ListBlank Γ
    l₁ l₂ : List Γ
    H : ∀ (i : Nat), Eq (l₁.getI i) (l₂.getI i)
    h✝ : LE.le l₁.length l₂.length
    i : Nat
    h : LT.lt i l₂.length
    h₂ : LT.lt i (HAppend.hAppend l₁ (List.replicate (HSub.hSub l₂.length l₁.lengt …
    ⊢ Eq (GetElem.getElem (HAppend.hAppend l₁ (List.replicate (HSub.hSub l₂.length …
  -/
  cases' lt_or_le i l₁.length with h' h'
    /-
      case refine_2.inl
      Γ : Type u_1
      i✝ : Inhabited Γ
      L₁ L₂ : Turing.ListBlank Γ
      l₁ l₂ : List Γ
      H : ∀ (i : Nat), Eq (l₁.getI i) (l₂.getI i)
      h✝ : LE.le l₁.length l₂.length
      i : Nat
      h : LT.lt i l₂.length
      h₂ : LT.lt i (HAppend.hAppend l₁ (List.replicate (HSub.hSub l₂.length l₁.lengt …
      h' : LT.lt i l₁.length
      ⊢ Eq (GetElem.getElem (HAppend.hAppend l₁ (List.replicate (HSub.hSub l₂.length …
    -/
  · simp [h', List.getElem_append _ h₂, ← List.getI_eq_getElem _ h, ← List.getI_eq_getElem _ h', H]
    /-
      🎉 no goals
    -/
  · rw [List.getElem_append_right h', List.getElem_replicate,
      ← List.getI_eq_default _ h', H, List.getI_eq_getElem _ h]


/-- Apply a function to a value stored at the nth position of the list. -/
@[simp]
def ListBlank.modifyNth {Γ} [Inhabited Γ] (f : Γ → Γ) : ℕ → ListBlank Γ → ListBlank Γ
  | 0, L => L.tail.cons (f L.head)
  | n + 1, L => (L.tail.modifyNth f n).cons L.head


theorem ListBlank.nth_modifyNth {Γ} [Inhabited Γ] (f : Γ → Γ) (n i) (L : ListBlank Γ) :
    (L.modifyNth f n).nth i = if i = n then f (L.nth i) else L.nth i := by
  /-
    Γ : Type u_1
    inst✝ : Inhabited Γ
    f : Γ → Γ
    n i : Nat
    L : Turing.ListBlank Γ
    ⊢ Eq ((Turing.ListBlank.modifyNth f n L).nth i) (ite (Eq i n) (f (L.nth i)) (L …
  -/
  induction' n with n IH generalizing i L
    /-
      case zero
      Γ : Type u_1
      inst✝ : Inhabited Γ
      f : Γ → Γ
      i : Nat
      L : Turing.ListBlank Γ
      ⊢ Eq ((Turing.ListBlank.modifyNth f 0 L).nth i) (ite (Eq i 0) (f (L.nth i)) (L …
    -/
  · cases i <;> simp only [ListBlank.nth_zero, if_true, ListBlank.head_cons, ListBlank.modifyNth,
      ListBlank.nth_succ, if_false, ListBlank.tail_cons, reduceCtorEq]
    /-
      case succ
      Γ : Type u_1
      inst✝ : Inhabited Γ
      f : Γ → Γ
      n : Nat
      IH : ∀ (i : Nat) (L : Turing.ListBlank Γ), Eq ((Turing.ListBlank.modifyNth f n …
      i : Nat
      L : Turing.ListBlank Γ
      ⊢ Eq ((Turing.ListBlank.modifyNth f (HAdd.hAdd n 1) L).nth i) (ite (Eq i (HAdd …
    -/
  · cases i
      /-
        case succ.zero
        Γ : Type u_1
        inst✝ : Inhabited Γ
        f : Γ → Γ
        n : Nat
        IH : ∀ (i : Nat) (L : Turing.ListBlank Γ), Eq ((Turing.ListBlank.modifyNth f n …
        L : Turing.ListBlank Γ
        ⊢ Eq ((Turing.ListBlank.modifyNth f (HAdd.hAdd n 1) L).nth 0) (ite (Eq 0 (HAdd …
      -/
    · rw [if_neg (Nat.succ_ne_zero _).symm]
      /-
        case succ.zero
        Γ : Type u_1
        inst✝ : Inhabited Γ
        f : Γ → Γ
        n : Nat
        IH : ∀ (i : Nat) (L : Turing.ListBlank Γ), Eq ((Turing.ListBlank.modifyNth f n …
        L : Turing.ListBlank Γ
        ⊢ Eq ((Turing.ListBlank.modifyNth f (HAdd.hAdd n 1) L).nth 0) (L.nth 0)
      -/
      simp only [ListBlank.nth_zero, ListBlank.head_cons, ListBlank.modifyNth]
      /-
        🎉 no goals
      -/
      /-
        case succ.succ
        Γ : Type u_1
        inst✝ : Inhabited Γ
        f : Γ → Γ
        n : Nat
        IH : ∀ (i : Nat) (L : Turing.ListBlank Γ), Eq ((Turing.ListBlank.modifyNth f n …
        L : Turing.ListBlank Γ
        n✝ : Nat
        ⊢ Eq ((Turing.ListBlank.modifyNth f (HAdd.hAdd n 1) L).nth (HAdd.hAdd n✝ 1)) ( …
      -/
    · simp only [IH, ListBlank.modifyNth, ListBlank.nth_succ, ListBlank.tail_cons, Nat.succ.injEq]
      /-
        🎉 no goals
      -/


/-- A pointed map of `Inhabited` types is a map that sends one default value to the other. -/
structure PointedMap.{u, v} (Γ : Type u) (Γ' : Type v) [Inhabited Γ] [Inhabited Γ'] :
    Type max u v where
  /-- The map underlying this instance. -/
  f : Γ → Γ'
  map_pt' : f default = default


instance {Γ Γ'} [Inhabited Γ] [Inhabited Γ'] : Inhabited (PointedMap Γ Γ') :=
  ⟨⟨default, rfl⟩⟩


instance {Γ Γ'} [Inhabited Γ] [Inhabited Γ'] : CoeFun (PointedMap Γ Γ') fun _ ↦ Γ → Γ' :=
  ⟨PointedMap.f⟩

-- @[simp] -- Porting note (https://github.com/leanprover-community/mathlib4/issues/10685): dsimp can prove this

theorem PointedMap.mk_val {Γ Γ'} [Inhabited Γ] [Inhabited Γ'] (f : Γ → Γ') (pt) :
    (PointedMap.mk f pt : Γ → Γ') = f :=
  rfl


@[simp]
theorem PointedMap.map_pt {Γ Γ'} [Inhabited Γ] [Inhabited Γ'] (f : PointedMap Γ Γ') :
    f default = default :=
  PointedMap.map_pt' _


@[simp]
theorem PointedMap.headI_map {Γ Γ'} [Inhabited Γ] [Inhabited Γ'] (f : PointedMap Γ Γ')
    (l : List Γ) : (l.map f).headI = f l.headI := by
  /-
    Γ : Type u_1
    Γ' : Type u_2
    inst✝¹ : Inhabited Γ
    inst✝ : Inhabited Γ'
    f : Turing.PointedMap Γ Γ'
    l : List Γ
    ⊢ Eq (List.map f.f l).headI (f.f l.headI)
  -/
  cases l <;> [exact (PointedMap.map_pt f).symm; rfl]
  /-
    🎉 no goals
  -/


/-- The `map` function on lists is well defined on `ListBlank`s provided that the map is
pointed. -/
def ListBlank.map {Γ Γ'} [Inhabited Γ] [Inhabited Γ'] (f : PointedMap Γ Γ') (l : ListBlank Γ) :
    ListBlank Γ' := by
  /-
    Γ : Type ?u.24619
    Γ' : Type ?u.24618
    inst✝¹ : Inhabited Γ
    inst✝ : Inhabited Γ'
    f : Turing.PointedMap Γ Γ'
    l : Turing.ListBlank Γ
    ⊢ Turing.ListBlank Γ'
  -/
  apply l.liftOn (fun l ↦ ListBlank.mk (List.map f l))
  /-
    Γ : Type ?u.24619
    Γ' : Type ?u.24618
    inst✝¹ : Inhabited Γ
    inst✝ : Inhabited Γ'
    f : Turing.PointedMap Γ Γ'
    l : Turing.ListBlank Γ
    ⊢ ∀ (a b : List Γ), Turing.BlankExtends a b → Eq (Turing.ListBlank.mk (List.ma …
  -/
  rintro l _ ⟨i, rfl⟩; refine Quotient.sound' (Or.inl ⟨i, ?_⟩)
  /-
    case intro
    Γ : Type ?u.24619
    Γ' : Type ?u.24618
    inst✝¹ : Inhabited Γ
    inst✝ : Inhabited Γ'
    f : Turing.PointedMap Γ Γ'
    l✝ : Turing.ListBlank Γ
    l : List Γ
    i : Nat
    ⊢ Eq (List.map f.f (HAppend.hAppend l (List.replicate i Inhabited.default))) ( …
  -/
  simp only [PointedMap.map_pt, List.map_append, List.map_replicate]
  /-
    🎉 no goals
  -/


@[simp]
theorem ListBlank.map_mk {Γ Γ'} [Inhabited Γ] [Inhabited Γ'] (f : PointedMap Γ Γ') (l : List Γ) :
    (ListBlank.mk l).map f = ListBlank.mk (l.map f) :=
  rfl


@[simp]
theorem ListBlank.head_map {Γ Γ'} [Inhabited Γ] [Inhabited Γ'] (f : PointedMap Γ Γ')
    (l : ListBlank Γ) : (l.map f).head = f l.head := by
  /-
    Γ : Type u_1
    Γ' : Type u_2
    inst✝¹ : Inhabited Γ
    inst✝ : Inhabited Γ'
    f : Turing.PointedMap Γ Γ'
    l : Turing.ListBlank Γ
    ⊢ Eq (Turing.ListBlank.map f l).head (f.f l.head)
  -/
  conv => lhs; rw [← ListBlank.cons_head_tail l]
  /-
    Γ : Type u_1
    Γ' : Type u_2
    inst✝¹ : Inhabited Γ
    inst✝ : Inhabited Γ'
    f : Turing.PointedMap Γ Γ'
    l : Turing.ListBlank Γ
    ⊢ Eq (Turing.ListBlank.map f (Turing.ListBlank.cons l.head l.tail)).head (f.f  …
  -/
  exact Quotient.inductionOn' l fun a ↦ rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem ListBlank.tail_map {Γ Γ'} [Inhabited Γ] [Inhabited Γ'] (f : PointedMap Γ Γ')
    (l : ListBlank Γ) : (l.map f).tail = l.tail.map f := by
  /-
    Γ : Type u_1
    Γ' : Type u_2
    inst✝¹ : Inhabited Γ
    inst✝ : Inhabited Γ'
    f : Turing.PointedMap Γ Γ'
    l : Turing.ListBlank Γ
    ⊢ Eq (Turing.ListBlank.map f l).tail (Turing.ListBlank.map f l.tail)
  -/
  conv => lhs; rw [← ListBlank.cons_head_tail l]
  /-
    Γ : Type u_1
    Γ' : Type u_2
    inst✝¹ : Inhabited Γ
    inst✝ : Inhabited Γ'
    f : Turing.PointedMap Γ Γ'
    l : Turing.ListBlank Γ
    ⊢ Eq (Turing.ListBlank.map f (Turing.ListBlank.cons l.head l.tail)).tail (Turi …
  -/
  exact Quotient.inductionOn' l fun a ↦ rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem ListBlank.map_cons {Γ Γ'} [Inhabited Γ] [Inhabited Γ'] (f : PointedMap Γ Γ')
    (l : ListBlank Γ) (a : Γ) : (l.cons a).map f = (l.map f).cons (f a) := by
  /-
    Γ : Type u_1
    Γ' : Type u_2
    inst✝¹ : Inhabited Γ
    inst✝ : Inhabited Γ'
    f : Turing.PointedMap Γ Γ'
    l : Turing.ListBlank Γ
    a : Γ
    ⊢ Eq (Turing.ListBlank.map f (Turing.ListBlank.cons a l)) (Turing.ListBlank.co …
  -/
  refine (ListBlank.cons_head_tail _).symm.trans ?_
  /-
    Γ : Type u_1
    Γ' : Type u_2
    inst✝¹ : Inhabited Γ
    inst✝ : Inhabited Γ'
    f : Turing.PointedMap Γ Γ'
    l : Turing.ListBlank Γ
    a : Γ
    ⊢ Eq (Turing.ListBlank.cons (Turing.ListBlank.map f (Turing.ListBlank.cons a l …
  -/
  simp only [ListBlank.head_map, ListBlank.head_cons, ListBlank.tail_map, ListBlank.tail_cons]
  /-
    🎉 no goals
  -/


@[simp]
theorem ListBlank.nth_map {Γ Γ'} [Inhabited Γ] [Inhabited Γ'] (f : PointedMap Γ Γ')
    (l : ListBlank Γ) (n : ℕ) : (l.map f).nth n = f (l.nth n) := by
  /-
    Γ : Type u_1
    Γ' : Type u_2
    inst✝¹ : Inhabited Γ
    inst✝ : Inhabited Γ'
    f : Turing.PointedMap Γ Γ'
    l : Turing.ListBlank Γ
    n : Nat
    ⊢ Eq ((Turing.ListBlank.map f l).nth n) (f.f (l.nth n))
  -/
  refine l.inductionOn fun l ↦ ?_
  -- Porting note: Added `suffices` to get `simp` to work.
  /-
    Γ : Type u_1
    Γ' : Type u_2
    inst✝¹ : Inhabited Γ
    inst✝ : Inhabited Γ'
    f : Turing.PointedMap Γ Γ'
    l✝ : Turing.ListBlank Γ
    n : Nat
    l : List Γ
    ⊢ Eq ((Turing.ListBlank.map f (Quotient.mk (Turing.BlankRel.setoid Γ) l)).nth  …
  -/
  suffices ((mk l).map f).nth n = f ((mk l).nth n) by exact this
  /-
    Γ : Type u_1
    Γ' : Type u_2
    inst✝¹ : Inhabited Γ
    inst✝ : Inhabited Γ'
    f : Turing.PointedMap Γ Γ'
    l✝ : Turing.ListBlank Γ
    n : Nat
    l : List Γ
    ⊢ Eq ((Turing.ListBlank.map f (Turing.ListBlank.mk l)).nth n) (f.f ((Turing.Li …
  -/
  simp only [ListBlank.map_mk, ListBlank.nth_mk, ← List.getD_default_eq_getI]
  /-
    Γ : Type u_1
    Γ' : Type u_2
    inst✝¹ : Inhabited Γ
    inst✝ : Inhabited Γ'
    f : Turing.PointedMap Γ Γ'
    l✝ : Turing.ListBlank Γ
    n : Nat
    l : List Γ
    ⊢ Eq ((List.map f.f l).getD n Inhabited.default) (f.f (l.getD n Inhabited.defa …
  -/
  rw [← List.getD_map _ _ f]
  /-
    Γ : Type u_1
    Γ' : Type u_2
    inst✝¹ : Inhabited Γ
    inst✝ : Inhabited Γ'
    f : Turing.PointedMap Γ Γ'
    l✝ : Turing.ListBlank Γ
    n : Nat
    l : List Γ
    ⊢ Eq ((List.map f.f l).getD n Inhabited.default) ((List.map f.f l).getD n (f.f …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- The `i`-th projection as a pointed map. -/
def proj {ι : Type*} {Γ : ι → Type*} [∀ i, Inhabited (Γ i)] (i : ι) :
    PointedMap (∀ i, Γ i) (Γ i) :=
  ⟨fun a ↦ a i, rfl⟩


theorem proj_map_nth {ι : Type*} {Γ : ι → Type*} [∀ i, Inhabited (Γ i)] (i : ι) (L n) :
    (ListBlank.map (@proj ι Γ _ i) L).nth n = L.nth n i := by
  /-
    ι : Type u_1
    Γ : ι → Type u_2
    inst✝ : (i : ι) → Inhabited (Γ i)
    i : ι
    L : Turing.ListBlank ((i : ι) → Γ i)
    n : Nat
    ⊢ Eq ((Turing.ListBlank.map (Turing.proj i) L).nth n) (L.nth n i)
  -/
  rw [ListBlank.nth_map]; rfl
                          /-
                            🎉 no goals
                          -/


theorem ListBlank.map_modifyNth {Γ Γ'} [Inhabited Γ] [Inhabited Γ'] (F : PointedMap Γ Γ')
    (f : Γ → Γ) (f' : Γ' → Γ') (H : ∀ x, F (f x) = f' (F x)) (n) (L : ListBlank Γ) :
    (L.modifyNth f n).map F = (L.map F).modifyNth f' n := by
  /-
    Γ : Type u_1
    Γ' : Type u_2
    inst✝¹ : Inhabited Γ
    inst✝ : Inhabited Γ'
    F : Turing.PointedMap Γ Γ'
    f : Γ → Γ
    f' : Γ' → Γ'
    H : ∀ (x : Γ), Eq (F.f (f x)) (f' (F.f x))
    n : Nat
    L : Turing.ListBlank Γ
    ⊢ Eq (Turing.ListBlank.map F (Turing.ListBlank.modifyNth f n L)) (Turing.ListB …
  -/
  induction' n with n IH generalizing L <;>
    /-
      case zero
      Γ : Type u_1
      Γ' : Type u_2
      inst✝¹ : Inhabited Γ
      inst✝ : Inhabited Γ'
      F : Turing.PointedMap Γ Γ'
      f : Γ → Γ
      f' : Γ' → Γ'
      H : ∀ (x : Γ), Eq (F.f (f x)) (f' (F.f x))
      L : Turing.ListBlank Γ
      ⊢ Eq (Turing.ListBlank.map F (Turing.ListBlank.modifyNth f 0 L)) (Turing.ListB …
    -/
    /-
      🎉 no goals
    -/
    simp only [*, ListBlank.head_map, ListBlank.modifyNth, ListBlank.map_cons, ListBlank.tail_map]
    /-
      🎉 no goals
    -/


/-- Append a list on the left side of a `ListBlank`. -/
@[simp]
def ListBlank.append {Γ} [Inhabited Γ] : List Γ → ListBlank Γ → ListBlank Γ
  | [], L => L
  | a :: l, L => ListBlank.cons a (ListBlank.append l L)


@[simp]
theorem ListBlank.append_mk {Γ} [Inhabited Γ] (l₁ l₂ : List Γ) :
    ListBlank.append l₁ (ListBlank.mk l₂) = ListBlank.mk (l₁ ++ l₂) := by
  /-
    Γ : Type u_1
    inst✝ : Inhabited Γ
    l₁ l₂ : List Γ
    ⊢ Eq (Turing.ListBlank.append l₁ (Turing.ListBlank.mk l₂)) (Turing.ListBlank.m …
  -/
  induction l₁ <;>
    /-
      case nil
      Γ : Type u_1
      inst✝ : Inhabited Γ
      l₂ : List Γ
      ⊢ Eq (Turing.ListBlank.append List.nil (Turing.ListBlank.mk l₂)) (Turing.ListB …
    -/
    /-
      🎉 no goals
    -/
    simp only [*, ListBlank.append, List.nil_append, List.cons_append, ListBlank.cons_mk]
    /-
      🎉 no goals
    -/


theorem ListBlank.append_assoc {Γ} [Inhabited Γ] (l₁ l₂ : List Γ) (l₃ : ListBlank Γ) :
    ListBlank.append (l₁ ++ l₂) l₃ = ListBlank.append l₁ (ListBlank.append l₂ l₃) := by
  /-
    Γ : Type u_1
    inst✝ : Inhabited Γ
    l₁ l₂ : List Γ
    l₃ : Turing.ListBlank Γ
    ⊢ Eq (Turing.ListBlank.append (HAppend.hAppend l₁ l₂) l₃) (Turing.ListBlank.ap …
  -/
  refine l₃.inductionOn fun l ↦ ?_
  -- Porting note: Added `suffices` to get `simp` to work.
  /-
    Γ : Type u_1
    inst✝ : Inhabited Γ
    l₁ l₂ : List Γ
    l₃ : Turing.ListBlank Γ
    l : List Γ
    ⊢ Eq (Turing.ListBlank.append (HAppend.hAppend l₁ l₂) (Quotient.mk (Turing.Bla …
  -/
  suffices append (l₁ ++ l₂) (mk l) = append l₁ (append l₂ (mk l)) by exact this
  /-
    Γ : Type u_1
    inst✝ : Inhabited Γ
    l₁ l₂ : List Γ
    l₃ : Turing.ListBlank Γ
    l : List Γ
    ⊢ Eq (Turing.ListBlank.append (HAppend.hAppend l₁ l₂) (Turing.ListBlank.mk l)) …
  -/
  simp only [ListBlank.append_mk, List.append_assoc]
  /-
    🎉 no goals
  -/


/-- The `flatMap` function on lists is well defined on `ListBlank`s provided that the default
element is sent to a sequence of default elements. -/
def ListBlank.flatMap {Γ Γ'} [Inhabited Γ] [Inhabited Γ'] (l : ListBlank Γ) (f : Γ → List Γ')
    (hf : ∃ n, f default = List.replicate n default) : ListBlank Γ' := by
  /-
    Γ : Type ?u.32652
    Γ' : Type ?u.32668
    inst✝¹ : Inhabited Γ
    inst✝ : Inhabited Γ'
    l : Turing.ListBlank Γ
    f : Γ → List Γ'
    hf : Exists fun n => Eq (f Inhabited.default) (List.replicate n Inhabited.defa …
    ⊢ Turing.ListBlank Γ'
  -/
  apply l.liftOn (fun l ↦ ListBlank.mk (List.flatMap l f))
  /-
    Γ : Type ?u.32652
    Γ' : Type ?u.32668
    inst✝¹ : Inhabited Γ
    inst✝ : Inhabited Γ'
    l : Turing.ListBlank Γ
    f : Γ → List Γ'
    hf : Exists fun n => Eq (f Inhabited.default) (List.replicate n Inhabited.defa …
    ⊢ ∀ (a b : List Γ), Turing.BlankExtends a b → Eq (Turing.ListBlank.mk (a.flatM …
  -/
  rintro l _ ⟨i, rfl⟩; cases' hf with n e; refine Quotient.sound' (Or.inl ⟨i * n, ?_⟩)
  /-
    case intro.intro
    Γ : Type ?u.32652
    Γ' : Type ?u.32668
    inst✝¹ : Inhabited Γ
    inst✝ : Inhabited Γ'
    l✝ : Turing.ListBlank Γ
    f : Γ → List Γ'
    l : List Γ
    i n : Nat
    e : Eq (f Inhabited.default) (List.replicate n Inhabited.default)
    ⊢ Eq ((HAppend.hAppend l (List.replicate i Inhabited.default)).flatMap f) (HAp …
  -/
  rw [List.flatMap_append, mul_comm]; congr
  /-
    case intro.intro.e_a
    Γ : Type ?u.32652
    Γ' : Type ?u.32668
    inst✝¹ : Inhabited Γ
    inst✝ : Inhabited Γ'
    l✝ : Turing.ListBlank Γ
    f : Γ → List Γ'
    l : List Γ
    i n : Nat
    e : Eq (f Inhabited.default) (List.replicate n Inhabited.default)
    ⊢ Eq ((List.replicate i Inhabited.default).flatMap f) (List.replicate (HMul.hM …
  -/
  induction' i with i IH
    /-
      case intro.intro.e_a.zero
      Γ : Type ?u.32652
      Γ' : Type ?u.32668
      inst✝¹ : Inhabited Γ
      inst✝ : Inhabited Γ'
      l✝ : Turing.ListBlank Γ
      f : Γ → List Γ'
      l : List Γ
      n : Nat
      e : Eq (f Inhabited.default) (List.replicate n Inhabited.default)
      ⊢ Eq ((List.replicate 0 Inhabited.default).flatMap f) (List.replicate (HMul.hM …
    -/
  · rfl
    /-
      🎉 no goals
    -/
  simp only [IH, e, List.replicate_add, Nat.mul_succ, add_comm, List.replicate_succ,
    List.flatMap_cons]


@[deprecated (since := "2024-10-16")] alias ListBlank.bind := ListBlank.flatMap


@[simp]
theorem ListBlank.flatMap_mk
    {Γ Γ'} [Inhabited Γ] [Inhabited Γ'] (l : List Γ) (f : Γ → List Γ') (hf) :
    (ListBlank.mk l).flatMap f hf = ListBlank.mk (l.flatMap f) :=
  rfl


@[deprecated (since := "2024-10-16")] alias ListBlank.bind_mk := ListBlank.flatMap_mk


@[simp]
theorem ListBlank.cons_flatMap {Γ Γ'} [Inhabited Γ] [Inhabited Γ'] (a : Γ) (l : ListBlank Γ)
    (f : Γ → List Γ') (hf) : (l.cons a).flatMap f hf = (l.flatMap f hf).append (f a) := by
  /-
    Γ : Type u_1
    Γ' : Type u_2
    inst✝¹ : Inhabited Γ
    inst✝ : Inhabited Γ'
    a : Γ
    l : Turing.ListBlank Γ
    f : Γ → List Γ'
    hf : Exists fun n => Eq (f Inhabited.default) (List.replicate n Inhabited.defa …
    ⊢ Eq ((Turing.ListBlank.cons a l).flatMap f hf) (Turing.ListBlank.append (f a) …
  -/
  refine l.inductionOn fun l ↦ ?_
  -- Porting note: Added `suffices` to get `simp` to work.
  /-
    Γ : Type u_1
    Γ' : Type u_2
    inst✝¹ : Inhabited Γ
    inst✝ : Inhabited Γ'
    a : Γ
    l✝ : Turing.ListBlank Γ
    f : Γ → List Γ'
    hf : Exists fun n => Eq (f Inhabited.default) (List.replicate n Inhabited.defa …
    l : List Γ
    ⊢ Eq ((Turing.ListBlank.cons a (Quotient.mk (Turing.BlankRel.setoid Γ) l)).fla …
  -/
  suffices ((mk l).cons a).flatMap f hf = ((mk l).flatMap f hf).append (f a) by exact this
  /-
    Γ : Type u_1
    Γ' : Type u_2
    inst✝¹ : Inhabited Γ
    inst✝ : Inhabited Γ'
    a : Γ
    l✝ : Turing.ListBlank Γ
    f : Γ → List Γ'
    hf : Exists fun n => Eq (f Inhabited.default) (List.replicate n Inhabited.defa …
    l : List Γ
    ⊢ Eq ((Turing.ListBlank.cons a (Turing.ListBlank.mk l)).flatMap f hf) (Turing. …
  -/
  simp only [ListBlank.append_mk, ListBlank.flatMap_mk, ListBlank.cons_mk, List.flatMap_cons]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-16")] alias ListBlank.cons_bind := ListBlank.cons_flatMap


/-- The tape of a Turing machine is composed of a head element (which we imagine to be the
current position of the head), together with two `ListBlank`s denoting the portions of the tape
going off to the left and right. When the Turing machine moves right, an element is pulled from the
right side and becomes the new head, while the head element is `cons`ed onto the left side. -/
structure Tape (Γ : Type*) [Inhabited Γ] where
  /-- The current position of the head. -/
  head : Γ
  /-- The portion of the tape going off to the left. -/
  left : ListBlank Γ
  /-- The portion of the tape going off to the right. -/
  right : ListBlank Γ


instance Tape.inhabited {Γ} [Inhabited Γ] : Inhabited (Tape Γ) :=
      /-
        Γ : Type ?u.36428
        inst✝ : Inhabited Γ
        ⊢ Turing.Tape Γ
      -/
                      /-
                        🎉 no goals
                      -/
                      /-
                        🎉 no goals
                      -/
  ⟨by constructor <;> apply default⟩
                      /-
                        🎉 no goals
                      -/


/-- A direction for the Turing machine `move` command, either
  left or right. -/
inductive Dir
  | left
  | right
  deriving DecidableEq, Inhabited


/-- The "inclusive" left side of the tape, including both `left` and `head`. -/
def Tape.left₀ {Γ} [Inhabited Γ] (T : Tape Γ) : ListBlank Γ :=
  T.left.cons T.head


/-- The "inclusive" right side of the tape, including both `right` and `head`. -/
def Tape.right₀ {Γ} [Inhabited Γ] (T : Tape Γ) : ListBlank Γ :=
  T.right.cons T.head


/-- Move the tape in response to a motion of the Turing machine. Note that `T.move Dir.left` makes
`T.left` smaller; the Turing machine is moving left and the tape is moving right. -/
def Tape.move {Γ} [Inhabited Γ] : Dir → Tape Γ → Tape Γ
  | Dir.left, ⟨a, L, R⟩ => ⟨L.head, L.tail, R.cons a⟩
  | Dir.right, ⟨a, L, R⟩ => ⟨R.head, L.cons a, R.tail⟩


@[simp]
theorem Tape.move_left_right {Γ} [Inhabited Γ] (T : Tape Γ) :
    (T.move Dir.left).move Dir.right = T := by
  /-
    Γ : Type u_1
    inst✝ : Inhabited Γ
    T : Turing.Tape Γ
    ⊢ Eq (Turing.Tape.move Turing.Dir.right (Turing.Tape.move Turing.Dir.left T)) T
  -/
  cases T; simp [Tape.move]
           /-
             🎉 no goals
           -/


@[simp]
theorem Tape.move_right_left {Γ} [Inhabited Γ] (T : Tape Γ) :
    (T.move Dir.right).move Dir.left = T := by
  /-
    Γ : Type u_1
    inst✝ : Inhabited Γ
    T : Turing.Tape Γ
    ⊢ Eq (Turing.Tape.move Turing.Dir.left (Turing.Tape.move Turing.Dir.right T)) T
  -/
  cases T; simp [Tape.move]
           /-
             🎉 no goals
           -/


/-- Construct a tape from a left side and an inclusive right side. -/
def Tape.mk' {Γ} [Inhabited Γ] (L R : ListBlank Γ) : Tape Γ :=
  ⟨R.head, L, R.tail⟩


@[simp]
theorem Tape.mk'_left {Γ} [Inhabited Γ] (L R : ListBlank Γ) : (Tape.mk' L R).left = L :=
  rfl


@[simp]
theorem Tape.mk'_head {Γ} [Inhabited Γ] (L R : ListBlank Γ) : (Tape.mk' L R).head = R.head :=
  rfl


@[simp]
theorem Tape.mk'_right {Γ} [Inhabited Γ] (L R : ListBlank Γ) : (Tape.mk' L R).right = R.tail :=
  rfl


@[simp]
theorem Tape.mk'_right₀ {Γ} [Inhabited Γ] (L R : ListBlank Γ) : (Tape.mk' L R).right₀ = R :=
  ListBlank.cons_head_tail _


@[simp]
theorem Tape.mk'_left_right₀ {Γ} [Inhabited Γ] (T : Tape Γ) : Tape.mk' T.left T.right₀ = T := by
  /-
    Γ : Type u_1
    inst✝ : Inhabited Γ
    T : Turing.Tape Γ
    ⊢ Eq (Turing.Tape.mk' T.left T.right₀) T
  -/
  cases T
  simp only [Tape.right₀, Tape.mk', ListBlank.head_cons, ListBlank.tail_cons, eq_self_iff_true,
    and_self_iff]


theorem Tape.exists_mk' {Γ} [Inhabited Γ] (T : Tape Γ) : ∃ L R, T = Tape.mk' L R :=
  ⟨_, _, (Tape.mk'_left_right₀ _).symm⟩


@[simp]
theorem Tape.move_left_mk' {Γ} [Inhabited Γ] (L R : ListBlank Γ) :
    (Tape.mk' L R).move Dir.left = Tape.mk' L.tail (R.cons L.head) := by
  simp only [Tape.move, Tape.mk', ListBlank.head_cons, eq_self_iff_true, ListBlank.cons_head_tail,
    and_self_iff, ListBlank.tail_cons]


@[simp]
theorem Tape.move_right_mk' {Γ} [Inhabited Γ] (L R : ListBlank Γ) :
    (Tape.mk' L R).move Dir.right = Tape.mk' (L.cons R.head) R.tail := by
  simp only [Tape.move, Tape.mk', ListBlank.head_cons, eq_self_iff_true, ListBlank.cons_head_tail,
    and_self_iff, ListBlank.tail_cons]


/-- Construct a tape from a left side and an inclusive right side. -/
def Tape.mk₂ {Γ} [Inhabited Γ] (L R : List Γ) : Tape Γ :=
  Tape.mk' (ListBlank.mk L) (ListBlank.mk R)


/-- Construct a tape from a list, with the head of the list at the TM head and the rest going
to the right. -/
def Tape.mk₁ {Γ} [Inhabited Γ] (l : List Γ) : Tape Γ :=
  Tape.mk₂ [] l


/-- The `nth` function of a tape is integer-valued, with index `0` being the head, negative indexes
on the left and positive indexes on the right. (Picture a number line.) -/
def Tape.nth {Γ} [Inhabited Γ] (T : Tape Γ) : ℤ → Γ
  | 0 => T.head
  | (n + 1 : ℕ) => T.right.nth n
  | -(n + 1 : ℕ) => T.left.nth n


@[simp]
theorem Tape.nth_zero {Γ} [Inhabited Γ] (T : Tape Γ) : T.nth 0 = T.1 :=
  rfl


theorem Tape.right₀_nth {Γ} [Inhabited Γ] (T : Tape Γ) (n : ℕ) : T.right₀.nth n = T.nth n := by
  /-
    Γ : Type u_1
    inst✝ : Inhabited Γ
    T : Turing.Tape Γ
    n : Nat
    ⊢ Eq (T.right₀.nth n) (T.nth ↑n)
  -/
  cases n <;> simp only [Tape.nth, Tape.right₀, Int.ofNat_zero, ListBlank.nth_zero,
    ListBlank.nth_succ, ListBlank.head_cons, ListBlank.tail_cons]


@[simp]
theorem Tape.mk'_nth_nat {Γ} [Inhabited Γ] (L R : ListBlank Γ) (n : ℕ) :
    (Tape.mk' L R).nth n = R.nth n := by
  /-
    Γ : Type u_1
    inst✝ : Inhabited Γ
    L R : Turing.ListBlank Γ
    n : Nat
    ⊢ Eq ((Turing.Tape.mk' L R).nth ↑n) (R.nth n)
  -/
  rw [← Tape.right₀_nth, Tape.mk'_right₀]
  /-
    🎉 no goals
  -/


@[simp]
theorem Tape.move_left_nth {Γ} [Inhabited Γ] :
    ∀ (T : Tape Γ) (i : ℤ), (T.move Dir.left).nth i = T.nth (i - 1)
  | ⟨_, _, _⟩, -(_ + 1 : ℕ) => (ListBlank.nth_succ _ _).symm
  | ⟨_, _, _⟩, 0 => (ListBlank.nth_zero _).symm
  | ⟨_, _, _⟩, 1 => (ListBlank.nth_zero _).trans (ListBlank.head_cons _ _)
  | ⟨a, L, R⟩, (n + 1 : ℕ) + 1 => by
    /-
      Γ : Type u_1
      inst✝ : Inhabited Γ
      a : Γ
      L R : Turing.ListBlank Γ
      n : Nat
      ⊢ Eq ((Turing.Tape.move Turing.Dir.left { head := a, left := L, right := R }). …
    -/
    rw [add_sub_cancel_right]
    /-
      Γ : Type u_1
      inst✝ : Inhabited Γ
      a : Γ
      L R : Turing.ListBlank Γ
      n : Nat
      ⊢ Eq ((Turing.Tape.move Turing.Dir.left { head := a, left := L, right := R }). …
    -/
    change (R.cons a).nth (n + 1) = R.nth n
    /-
      Γ : Type u_1
      inst✝ : Inhabited Γ
      a : Γ
      L R : Turing.ListBlank Γ
      n : Nat
      ⊢ Eq ((Turing.ListBlank.cons a R).nth (HAdd.hAdd n 1)) (R.nth n)
    -/
    rw [ListBlank.nth_succ, ListBlank.tail_cons]
    /-
      🎉 no goals
    -/


@[simp]
theorem Tape.move_right_nth {Γ} [Inhabited Γ] (T : Tape Γ) (i : ℤ) :
    (T.move Dir.right).nth i = T.nth (i + 1) := by
  /-
    Γ : Type u_1
    inst✝ : Inhabited Γ
    T : Turing.Tape Γ
    i : Int
    ⊢ Eq ((Turing.Tape.move Turing.Dir.right T).nth i) (T.nth (HAdd.hAdd i 1))
  -/
  conv => rhs; rw [← T.move_right_left]
  /-
    Γ : Type u_1
    inst✝ : Inhabited Γ
    T : Turing.Tape Γ
    i : Int
    ⊢ Eq ((Turing.Tape.move Turing.Dir.right T).nth i) ((Turing.Tape.move Turing.D …
  -/
  rw [Tape.move_left_nth, add_sub_cancel_right]
  /-
    🎉 no goals
  -/


@[simp]
theorem Tape.move_right_n_head {Γ} [Inhabited Γ] (T : Tape Γ) (i : ℕ) :
    ((Tape.move Dir.right)^[i] T).head = T.nth i := by
  /-
    Γ : Type u_1
    inst✝ : Inhabited Γ
    T : Turing.Tape Γ
    i : Nat
    ⊢ Eq (Nat.iterate (Turing.Tape.move Turing.Dir.right) i T).head (T.nth ↑i)
  -/
  induction i generalizing T
    /-
      case zero
      Γ : Type u_1
      inst✝ : Inhabited Γ
      T : Turing.Tape Γ
      ⊢ Eq (Nat.iterate (Turing.Tape.move Turing.Dir.right) 0 T).head (T.nth ↑0)
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case succ
      Γ : Type u_1
      inst✝ : Inhabited Γ
      n✝ : Nat
      a✝ : ∀ (T : Turing.Tape Γ), Eq (Nat.iterate (Turing.Tape.move Turing.Dir.right …
      T : Turing.Tape Γ
      ⊢ Eq (Nat.iterate (Turing.Tape.move Turing.Dir.right) (HAdd.hAdd n✝ 1) T).head …
    -/
  · simp only [*, Tape.move_right_nth, Int.ofNat_succ, iterate_succ, Function.comp_apply]
    /-
      🎉 no goals
    -/


/-- Replace the current value of the head on the tape. -/
def Tape.write {Γ} [Inhabited Γ] (b : Γ) (T : Tape Γ) : Tape Γ :=
  { T with head := b }


@[simp]
theorem Tape.write_self {Γ} [Inhabited Γ] : ∀ T : Tape Γ, T.write T.1 = T := by
  /-
    Γ : Type u_1
    inst✝ : Inhabited Γ
    ⊢ ∀ (T : Turing.Tape Γ), Eq (Turing.Tape.write T.head T) T
  -/
  rintro ⟨⟩; rfl
             /-
               🎉 no goals
             -/


@[simp]
theorem Tape.write_nth {Γ} [Inhabited Γ] (b : Γ) :
    ∀ (T : Tape Γ) {i : ℤ}, (T.write b).nth i = if i = 0 then b else T.nth i
  | _, 0 => rfl
  | _, (_ + 1 : ℕ) => rfl
  | _, -(_ + 1 : ℕ) => rfl


@[simp]
theorem Tape.write_mk' {Γ} [Inhabited Γ] (a b : Γ) (L R : ListBlank Γ) :
    (Tape.mk' L (R.cons a)).write b = Tape.mk' L (R.cons b) := by
  simp only [Tape.write, Tape.mk', ListBlank.head_cons, ListBlank.tail_cons, eq_self_iff_true,
    and_self_iff]


/-- Apply a pointed map to a tape to change the alphabet. -/
def Tape.map {Γ Γ'} [Inhabited Γ] [Inhabited Γ'] (f : PointedMap Γ Γ') (T : Tape Γ) : Tape Γ' :=
  ⟨f T.1, T.2.map f, T.3.map f⟩


@[simp]
theorem Tape.map_fst {Γ Γ'} [Inhabited Γ] [Inhabited Γ'] (f : PointedMap Γ Γ') :
    ∀ T : Tape Γ, (T.map f).1 = f T.1 := by
  /-
    Γ : Type u_1
    Γ' : Type u_2
    inst✝¹ : Inhabited Γ
    inst✝ : Inhabited Γ'
    f : Turing.PointedMap Γ Γ'
    ⊢ ∀ (T : Turing.Tape Γ), Eq (Turing.Tape.map f T).head (f.f T.head)
  -/
  rintro ⟨⟩; rfl
             /-
               🎉 no goals
             -/


@[simp]
theorem Tape.map_write {Γ Γ'} [Inhabited Γ] [Inhabited Γ'] (f : PointedMap Γ Γ') (b : Γ) :
    ∀ T : Tape Γ, (T.write b).map f = (T.map f).write (f b) := by
  /-
    Γ : Type u_1
    Γ' : Type u_2
    inst✝¹ : Inhabited Γ
    inst✝ : Inhabited Γ'
    f : Turing.PointedMap Γ Γ'
    b : Γ
    ⊢ ∀ (T : Turing.Tape Γ), Eq (Turing.Tape.map f (Turing.Tape.write b T)) (Turin …
  -/
  rintro ⟨⟩; rfl
             /-
               🎉 no goals
             -/

-- Porting note: `simpNF` complains about LHS does not simplify when using the simp lemma on
--               itself, but it does indeed.

@[simp, nolint simpNF]
theorem Tape.write_move_right_n {Γ} [Inhabited Γ] (f : Γ → Γ) (L R : ListBlank Γ) (n : ℕ) :
    ((Tape.move Dir.right)^[n] (Tape.mk' L R)).write (f (R.nth n)) =
      (Tape.move Dir.right)^[n] (Tape.mk' L (R.modifyNth f n)) := by
  /-
    Γ : Type u_1
    inst✝ : Inhabited Γ
    f : Γ → Γ
    L R : Turing.ListBlank Γ
    n : Nat
    ⊢ Eq (Turing.Tape.write (f (R.nth n)) (Nat.iterate (Turing.Tape.move Turing.Di …
  -/
  induction' n with n IH generalizing L R
    /-
      case zero
      Γ : Type u_1
      inst✝ : Inhabited Γ
      f : Γ → Γ
      L R : Turing.ListBlank Γ
      ⊢ Eq (Turing.Tape.write (f (R.nth 0)) (Nat.iterate (Turing.Tape.move Turing.Di …
    -/
  · simp only [ListBlank.nth_zero, ListBlank.modifyNth, iterate_zero_apply]
    /-
      case zero
      Γ : Type u_1
      inst✝ : Inhabited Γ
      f : Γ → Γ
      L R : Turing.ListBlank Γ
      ⊢ Eq (Turing.Tape.write (f R.head) (Turing.Tape.mk' L R)) (Turing.Tape.mk' L ( …
    -/
    rw [← Tape.write_mk', ListBlank.cons_head_tail]
    /-
      🎉 no goals
    -/
  simp only [ListBlank.head_cons, ListBlank.nth_succ, ListBlank.modifyNth, Tape.move_right_mk',
    ListBlank.tail_cons, iterate_succ_apply, IH]


theorem Tape.map_move {Γ Γ'} [Inhabited Γ] [Inhabited Γ'] (f : PointedMap Γ Γ') (T : Tape Γ) (d) :
    (T.move d).map f = (T.map f).move d := by
  /-
    Γ : Type u_1
    Γ' : Type u_2
    inst✝¹ : Inhabited Γ
    inst✝ : Inhabited Γ'
    f : Turing.PointedMap Γ Γ'
    T : Turing.Tape Γ
    d : Turing.Dir
    ⊢ Eq (Turing.Tape.map f (Turing.Tape.move d T)) (Turing.Tape.move d (Turing.Ta …
  -/
  cases T
  /-
    case mk
    Γ : Type u_1
    Γ' : Type u_2
    inst✝¹ : Inhabited Γ
    inst✝ : Inhabited Γ'
    f : Turing.PointedMap Γ Γ'
    d : Turing.Dir
    head✝ : Γ
    left✝ right✝ : Turing.ListBlank Γ
    ⊢ Eq (Turing.Tape.map f (Turing.Tape.move d { head := head✝, left := left✝, ri …
  -/
  cases d <;> simp only [Tape.move, Tape.map, ListBlank.head_map, eq_self_iff_true,
    ListBlank.map_cons, and_self_iff, ListBlank.tail_map]


theorem Tape.map_mk' {Γ Γ'} [Inhabited Γ] [Inhabited Γ'] (f : PointedMap Γ Γ') (L R : ListBlank Γ) :
    (Tape.mk' L R).map f = Tape.mk' (L.map f) (R.map f) := by
  simp only [Tape.mk', Tape.map, ListBlank.head_map, eq_self_iff_true, and_self_iff,
    ListBlank.tail_map]


theorem Tape.map_mk₂ {Γ Γ'} [Inhabited Γ] [Inhabited Γ'] (f : PointedMap Γ Γ') (L R : List Γ) :
    (Tape.mk₂ L R).map f = Tape.mk₂ (L.map f) (R.map f) := by
  /-
    Γ : Type u_1
    Γ' : Type u_2
    inst✝¹ : Inhabited Γ
    inst✝ : Inhabited Γ'
    f : Turing.PointedMap Γ Γ'
    L R : List Γ
    ⊢ Eq (Turing.Tape.map f (Turing.Tape.mk₂ L R)) (Turing.Tape.mk₂ (List.map f.f  …
  -/
  simp only [Tape.mk₂, Tape.map_mk', ListBlank.map_mk]
  /-
    🎉 no goals
  -/


theorem Tape.map_mk₁ {Γ Γ'} [Inhabited Γ] [Inhabited Γ'] (f : PointedMap Γ Γ') (l : List Γ) :
    (Tape.mk₁ l).map f = Tape.mk₁ (l.map f) :=
  Tape.map_mk₂ _ _ _


/-- Run a state transition function `σ → Option σ` "to completion". The return value is the last
state returned before a `none` result. If the state transition function always returns `some`,
then the computation diverges, returning `Part.none`. -/
def eval {σ} (f : σ → Option σ) : σ → Part σ :=
  PFun.fix fun s ↦ Part.some <| (f s).elim (Sum.inl s) Sum.inr


/-- The reflexive transitive closure of a state transition function. `Reaches f a b` means
there is a finite sequence of steps `f a = some a₁`, `f a₁ = some a₂`, ... such that `aₙ = b`.
This relation permits zero steps of the state transition function. -/
def Reaches {σ} (f : σ → Option σ) : σ → σ → Prop :=
  ReflTransGen fun a b ↦ b ∈ f a


/-- The transitive closure of a state transition function. `Reaches₁ f a b` means there is a
nonempty finite sequence of steps `f a = some a₁`, `f a₁ = some a₂`, ... such that `aₙ = b`.
This relation does not permit zero steps of the state transition function. -/
def Reaches₁ {σ} (f : σ → Option σ) : σ → σ → Prop :=
  TransGen fun a b ↦ b ∈ f a


theorem reaches₁_eq {σ} {f : σ → Option σ} {a b c} (h : f a = f b) :
    Reaches₁ f a c ↔ Reaches₁ f b c :=
                                                           /-
                                                             σ : Type u_1
                                                             f : σ → Option σ
                                                             a b c : σ
                                                             h : Eq (f a) (f b)
                                                             ⊢ Iff (Exists fun b_1 => And (Membership.mem (f b) b_1) (Relation.ReflTransGen …
                                                           -/
  TransGen.head'_iff.trans (TransGen.head'_iff.trans <| by rw [h]).symm
                                                           /-
                                                             🎉 no goals
                                                           -/


theorem reaches_total {σ} {f : σ → Option σ} {a b c} (hab : Reaches f a b) (hac : Reaches f a c) :
    Reaches f b c ∨ Reaches f c b :=
  ReflTransGen.total_of_right_unique (fun _ _ _ ↦ Option.mem_unique) hab hac


theorem reaches₁_fwd {σ} {f : σ → Option σ} {a b c} (h₁ : Reaches₁ f a c) (h₂ : b ∈ f a) :
    Reaches f b c := by
  /-
    σ : Type u_1
    f : σ → Option σ
    a b c : σ
    h₁ : Turing.Reaches₁ f a c
    h₂ : Membership.mem (f a) b
    ⊢ Turing.Reaches f b c
  -/
  rcases TransGen.head'_iff.1 h₁ with ⟨b', hab, hbc⟩
  /-
    case intro.intro
    σ : Type u_1
    f : σ → Option σ
    a b c : σ
    h₁ : Turing.Reaches₁ f a c
    h₂ : Membership.mem (f a) b
    b' : σ
    hab : Membership.mem (f a) b'
    hbc : Relation.ReflTransGen (fun a b => Membership.mem (f a) b) b' c
    ⊢ Turing.Reaches f b c
  -/
  cases Option.mem_unique hab h₂; exact hbc
                                  /-
                                    🎉 no goals
                                  -/


/-- A variation on `Reaches`. `Reaches₀ f a b` holds if whenever `Reaches₁ f b c` then
`Reaches₁ f a c`. This is a weaker property than `Reaches` and is useful for replacing states with
equivalent states without taking a step. -/
def Reaches₀ {σ} (f : σ → Option σ) (a b : σ) : Prop :=
  ∀ c, Reaches₁ f b c → Reaches₁ f a c


theorem Reaches₀.trans {σ} {f : σ → Option σ} {a b c : σ} (h₁ : Reaches₀ f a b)
    (h₂ : Reaches₀ f b c) : Reaches₀ f a c
  | _, h₃ => h₁ _ (h₂ _ h₃)


@[refl]
theorem Reaches₀.refl {σ} {f : σ → Option σ} (a : σ) : Reaches₀ f a a
  | _, h => h


theorem Reaches₀.single {σ} {f : σ → Option σ} {a b : σ} (h : b ∈ f a) : Reaches₀ f a b
  | _, h₂ => h₂.head h


theorem Reaches₀.head {σ} {f : σ → Option σ} {a b c : σ} (h : b ∈ f a) (h₂ : Reaches₀ f b c) :
    Reaches₀ f a c :=
  (Reaches₀.single h).trans h₂


theorem Reaches₀.tail {σ} {f : σ → Option σ} {a b c : σ} (h₁ : Reaches₀ f a b) (h : c ∈ f b) :
    Reaches₀ f a c :=
  h₁.trans (Reaches₀.single h)


theorem reaches₀_eq {σ} {f : σ → Option σ} {a b} (e : f a = f b) : Reaches₀ f a b
  | _, h => (reaches₁_eq e).2 h


theorem Reaches₁.to₀ {σ} {f : σ → Option σ} {a b : σ} (h : Reaches₁ f a b) : Reaches₀ f a b
  | _, h₂ => h.trans h₂


theorem Reaches.to₀ {σ} {f : σ → Option σ} {a b : σ} (h : Reaches f a b) : Reaches₀ f a b
  | _, h₂ => h₂.trans_right h


theorem Reaches₀.tail' {σ} {f : σ → Option σ} {a b c : σ} (h : Reaches₀ f a b) (h₂ : c ∈ f b) :
    Reaches₁ f a c :=
  h _ (TransGen.single h₂)


/-- (co-)Induction principle for `eval`. If a property `C` holds of any point `a` evaluating to `b`
which is either terminal (meaning `a = b`) or where the next point also satisfies `C`, then it
holds of any point where `eval f a` evaluates to `b`. This formalizes the notion that if
`eval f a` evaluates to `b` then it reaches terminal state `b` in finitely many steps. -/
@[elab_as_elim]
def evalInduction {σ} {f : σ → Option σ} {b : σ} {C : σ → Sort*} {a : σ}
    (h : b ∈ eval f a) (H : ∀ a, b ∈ eval f a → (∀ a', f a = some a' → C a') → C a) : C a :=
  PFun.fixInduction h fun a' ha' h' ↦
                                                         /-
                                                           σ : Type ?u.60158
                                                           f : σ → Option σ
                                                           b : σ
                                                           C : σ → Sort u_1
                                                           a : σ
                                                           h : Membership.mem (Turing.eval f a) b
                                                           H : (a : σ) → Membership.mem (Turing.eval f a) b → ((a' : σ) → Eq (f a) (Optio …
                                                           a' : σ
                                                           ha' : Membership.mem (PFun.fix (fun s => Part.some ((f s).elim (Sum.inl s) Sum …
                                                           h' : (a'' : σ) → Membership.mem (Part.some ((f a').elim (Sum.inl a') Sum.inr)) …
                                                           b' : σ
                                                           e : Eq (f a') (Option.some b')
                                                           ⊢ Eq (Sum.inr b') ((f a').elim (Sum.inl a') Sum.inr)
                                                         -/
    H _ ha' fun b' e ↦ h' _ <| Part.mem_some_iff.2 <| by rw [e]; rfl
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


theorem mem_eval {σ} {f : σ → Option σ} {a b} : b ∈ eval f a ↔ Reaches f a b ∧ f b = none := by
  /-
    σ : Type u_1
    f : σ → Option σ
    a b : σ
    ⊢ Iff (Membership.mem (Turing.eval f a) b) (And (Turing.Reaches f a b) (Eq (f  …
  -/
  refine ⟨fun h ↦ ?_, fun ⟨h₁, h₂⟩ ↦ ?_⟩
  · -- Porting note: Explicitly specify `c`.
    /-
      case refine_1
      σ : Type u_1
      f : σ → Option σ
      a b : σ
      h : Membership.mem (Turing.eval f a) b
      ⊢ And (Turing.Reaches f a b) (Eq (f b) Option.none)
    -/
    refine @evalInduction _ _ _ (fun a ↦ Reaches f a b ∧ f b = none) _ h fun a h IH ↦ ?_
    /-
      case refine_1
      σ : Type u_1
      f : σ → Option σ
      a✝ b : σ
      h✝ : Membership.mem (Turing.eval f a✝) b
      a : σ
      h : Membership.mem (Turing.eval f a) b
      IH : ∀ (a' : σ), Eq (f a) (Option.some a') → (fun a => And (Turing.Reaches f a …
      ⊢ (fun a => And (Turing.Reaches f a b) (Eq (f b) Option.none)) a
    -/
    cases' e : f a with a'
    · rw [Part.mem_unique h
          (PFun.mem_fix_iff.2 <| Or.inl <| Part.mem_some_iff.2 <| by rw [e]; rfl)]
      /-
        case refine_1.none
        σ : Type u_1
        f : σ → Option σ
        a✝ b : σ
        h✝ : Membership.mem (Turing.eval f a✝) b
        a : σ
        h : Membership.mem (Turing.eval f a) b
        IH : ∀ (a' : σ), Eq (f a) (Option.some a') → (fun a => And (Turing.Reaches f a …
        e : Eq (f a) Option.none
        ⊢ And (Turing.Reaches f a a) (Eq (f a) Option.none)
      -/
      exact ⟨ReflTransGen.refl, e⟩
      /-
        🎉 no goals
      -/
      /-
        case refine_1.some
        σ : Type u_1
        f : σ → Option σ
        a✝ b : σ
        h✝ : Membership.mem (Turing.eval f a✝) b
        a : σ
        h : Membership.mem (Turing.eval f a) b
        IH : ∀ (a' : σ), Eq (f a) (Option.some a') → (fun a => And (Turing.Reaches f a …
        a' : σ
        e : Eq (f a) (Option.some a')
        ⊢ And (Turing.Reaches f a b) (Eq (f b) Option.none)
      -/
    · rcases PFun.mem_fix_iff.1 h with (h | ⟨_, h, _⟩) <;> rw [e] at h <;>
        /-
          case refine_1.some.inl
          σ : Type u_1
          f : σ → Option σ
          a✝ b : σ
          h✝¹ : Membership.mem (Turing.eval f a✝) b
          a : σ
          h✝ : Membership.mem (Turing.eval f a) b
          IH : ∀ (a' : σ), Eq (f a) (Option.some a') → (fun a => And (Turing.Reaches f a …
          a' : σ
          e : Eq (f a) (Option.some a')
          h : Membership.mem (Part.some ((Option.some a').elim (Sum.inl a) Sum.inr)) (Su …
          ⊢ And (Turing.Reaches f a b) (Eq (f b) Option.none)
        -/
        /-
          🎉 no goals
        -/
        cases Part.mem_some_iff.1 h
      /-
        case refine_1.some.inr.intro.intro.refl
        σ : Type u_1
        f : σ → Option σ
        a✝ b : σ
        h✝¹ : Membership.mem (Turing.eval f a✝) b
        a : σ
        h✝ : Membership.mem (Turing.eval f a) b
        IH : ∀ (a' : σ), Eq (f a) (Option.some a') → (fun a => And (Turing.Reaches f a …
        a' : σ
        e : Eq (f a) (Option.some a')
        h : Membership.mem (Part.some ((Option.some a').elim (Sum.inl a) Sum.inr)) (Su …
        right✝ : Membership.mem (PFun.fix (fun s => Part.some ((f s).elim (Sum.inl s)  …
        ⊢ And (Turing.Reaches f a b) (Eq (f b) Option.none)
      -/
      cases' IH a' e with h₁ h₂
      /-
        case refine_1.some.inr.intro.intro.refl.intro
        σ : Type u_1
        f : σ → Option σ
        a✝ b : σ
        h✝¹ : Membership.mem (Turing.eval f a✝) b
        a : σ
        h✝ : Membership.mem (Turing.eval f a) b
        IH : ∀ (a' : σ), Eq (f a) (Option.some a') → (fun a => And (Turing.Reaches f a …
        a' : σ
        e : Eq (f a) (Option.some a')
        h : Membership.mem (Part.some ((Option.some a').elim (Sum.inl a) Sum.inr)) (Su …
        right✝ : Membership.mem (PFun.fix (fun s => Part.some ((f s).elim (Sum.inl s)  …
        h₁ : Turing.Reaches f a' b
        h₂ : Eq (f b) Option.none
        ⊢ And (Turing.Reaches f a b) (Eq (f b) Option.none)
      -/
      exact ⟨ReflTransGen.head e h₁, h₂⟩
      /-
        🎉 no goals
      -/
    /-
      case refine_2
      σ : Type u_1
      f : σ → Option σ
      a b : σ
      x✝ : And (Turing.Reaches f a b) (Eq (f b) Option.none)
      h₁ : Turing.Reaches f a b
      h₂ : Eq (f b) Option.none
      ⊢ Membership.mem (Turing.eval f a) b
    -/
  · refine ReflTransGen.head_induction_on h₁ ?_ fun h _ IH ↦ ?_
      /-
        case refine_2.refine_1
        σ : Type u_1
        f : σ → Option σ
        a b : σ
        x✝ : And (Turing.Reaches f a b) (Eq (f b) Option.none)
        h₁ : Turing.Reaches f a b
        h₂ : Eq (f b) Option.none
        ⊢ Membership.mem (Turing.eval f b) b
      -/
    · refine PFun.mem_fix_iff.2 (Or.inl ?_)
      /-
        case refine_2.refine_1
        σ : Type u_1
        f : σ → Option σ
        a b : σ
        x✝ : And (Turing.Reaches f a b) (Eq (f b) Option.none)
        h₁ : Turing.Reaches f a b
        h₂ : Eq (f b) Option.none
        ⊢ Membership.mem (Part.some ((f b).elim (Sum.inl b) Sum.inr)) (Sum.inl b)
      -/
      rw [h₂]
      /-
        case refine_2.refine_1
        σ : Type u_1
        f : σ → Option σ
        a b : σ
        x✝ : And (Turing.Reaches f a b) (Eq (f b) Option.none)
        h₁ : Turing.Reaches f a b
        h₂ : Eq (f b) Option.none
        ⊢ Membership.mem (Part.some (Option.none.elim (Sum.inl b) Sum.inr)) (Sum.inl b)
      -/
      apply Part.mem_some
      /-
        🎉 no goals
      -/
      /-
        case refine_2.refine_2
        σ : Type u_1
        f : σ → Option σ
        a b : σ
        x✝¹ : And (Turing.Reaches f a b) (Eq (f b) Option.none)
        h₁ : Turing.Reaches f a b
        h₂ : Eq (f b) Option.none
        a✝ c✝ : σ
        h : Membership.mem (f a✝) c✝
        x✝ : Relation.ReflTransGen (fun a b => Membership.mem (f a) b) c✝ b
        IH : Membership.mem (Turing.eval f c✝) b
        ⊢ Membership.mem (Turing.eval f a✝) b
      -/
    · refine PFun.mem_fix_iff.2 (Or.inr ⟨_, ?_, IH⟩)
      /-
        case refine_2.refine_2
        σ : Type u_1
        f : σ → Option σ
        a b : σ
        x✝¹ : And (Turing.Reaches f a b) (Eq (f b) Option.none)
        h₁ : Turing.Reaches f a b
        h₂ : Eq (f b) Option.none
        a✝ c✝ : σ
        h : Membership.mem (f a✝) c✝
        x✝ : Relation.ReflTransGen (fun a b => Membership.mem (f a) b) c✝ b
        IH : Membership.mem (Turing.eval f c✝) b
        ⊢ Membership.mem (Part.some ((f a✝).elim (Sum.inl a✝) Sum.inr)) (Sum.inr c✝)
      -/
      rw [h]
      /-
        case refine_2.refine_2
        σ : Type u_1
        f : σ → Option σ
        a b : σ
        x✝¹ : And (Turing.Reaches f a b) (Eq (f b) Option.none)
        h₁ : Turing.Reaches f a b
        h₂ : Eq (f b) Option.none
        a✝ c✝ : σ
        h : Membership.mem (f a✝) c✝
        x✝ : Relation.ReflTransGen (fun a b => Membership.mem (f a) b) c✝ b
        IH : Membership.mem (Turing.eval f c✝) b
        ⊢ Membership.mem (Part.some ((Option.some c✝).elim (Sum.inl a✝) Sum.inr)) (Sum …
      -/
      apply Part.mem_some
      /-
        🎉 no goals
      -/


theorem eval_maximal₁ {σ} {f : σ → Option σ} {a b} (h : b ∈ eval f a) (c) : ¬Reaches₁ f b c
  | bc => by
    /-
      σ : Type u_1
      f : σ → Option σ
      a b : σ
      h : Membership.mem (Turing.eval f a) b
      c : σ
      x✝ : Turing.Reaches₁ f b c
      bc : Turing.Reaches₁ f b c := x✝
      ⊢ False
    -/
    let ⟨_, b0⟩ := mem_eval.1 h
    /-
      σ : Type u_1
      f : σ → Option σ
      a b : σ
      h : Membership.mem (Turing.eval f a) b
      c : σ
      x✝ : Turing.Reaches₁ f b c
      bc : Turing.Reaches₁ f b c := x✝
      left✝ : Turing.Reaches f a b
      b0 : Eq (f b) Option.none
      ⊢ False
    -/
    let ⟨b', h', _⟩ := TransGen.head'_iff.1 bc
    /-
      σ : Type u_1
      f : σ → Option σ
      a b : σ
      h : Membership.mem (Turing.eval f a) b
      c : σ
      x✝ : Turing.Reaches₁ f b c
      bc : Turing.Reaches₁ f b c := x✝
      left✝ : Turing.Reaches f a b
      b0 : Eq (f b) Option.none
      b' : σ
      h' : Membership.mem (f b) b'
      right✝ : Relation.ReflTransGen (fun a b => Membership.mem (f a) b) b' c
      ⊢ False
    -/
    cases b0.symm.trans h'
    /-
      🎉 no goals
    -/


theorem eval_maximal {σ} {f : σ → Option σ} {a b} (h : b ∈ eval f a) {c} : Reaches f b c ↔ c = b :=
  let ⟨_, b0⟩ := mem_eval.1 h
                                     /-
                                       σ : Type u_1
                                       f : σ → Option σ
                                       a b : σ
                                       h : Membership.mem (Turing.eval f a) b
                                       c : σ
                                       left✝ : Turing.Reaches f a b
                                       b0 : Eq (f b) Option.none
                                       b' : σ
                                       h' : Membership.mem (f b) b'
                                       ⊢ False
                                     -/
  reflTransGen_iff_eq fun b' h' ↦ by cases b0.symm.trans h'
                                     /-
                                       🎉 no goals
                                     -/


theorem reaches_eval {σ} {f : σ → Option σ} {a b} (ab : Reaches f a b) : eval f a = eval f b := by
  /-
    σ : Type u_1
    f : σ → Option σ
    a b : σ
    ab : Turing.Reaches f a b
    ⊢ Eq (Turing.eval f a) (Turing.eval f b)
  -/
  refine Part.ext fun _ ↦ ⟨fun h ↦ ?_, fun h ↦ ?_⟩
    /-
      case refine_1
      σ : Type u_1
      f : σ → Option σ
      a b : σ
      ab : Turing.Reaches f a b
      x✝ : σ
      h : Membership.mem (Turing.eval f a) x✝
      ⊢ Membership.mem (Turing.eval f b) x✝
    -/
  · have ⟨ac, c0⟩ := mem_eval.1 h
    exact mem_eval.2 ⟨(or_iff_left_of_imp fun cb ↦ (eval_maximal h).1 cb ▸ ReflTransGen.refl).1
      (reaches_total ab ac), c0⟩
    /-
      case refine_2
      σ : Type u_1
      f : σ → Option σ
      a b : σ
      ab : Turing.Reaches f a b
      x✝ : σ
      h : Membership.mem (Turing.eval f b) x✝
      ⊢ Membership.mem (Turing.eval f a) x✝
    -/
  · have ⟨bc, c0⟩ := mem_eval.1 h
    /-
      case refine_2
      σ : Type u_1
      f : σ → Option σ
      a b : σ
      ab : Turing.Reaches f a b
      x✝ : σ
      h : Membership.mem (Turing.eval f b) x✝
      bc : Turing.Reaches f b x✝
      c0 : Eq (f x✝) Option.none
      ⊢ Membership.mem (Turing.eval f a) x✝
    -/
    exact mem_eval.2 ⟨ab.trans bc, c0⟩
    /-
      🎉 no goals
    -/


/-- Given a relation `tr : σ₁ → σ₂ → Prop` between state spaces, and state transition functions
`f₁ : σ₁ → Option σ₁` and `f₂ : σ₂ → Option σ₂`, `Respects f₁ f₂ tr` means that if `tr a₁ a₂` holds
initially and `f₁` takes a step to `a₂` then `f₂` will take one or more steps before reaching a
state `b₂` satisfying `tr a₂ b₂`, and if `f₁ a₁` terminates then `f₂ a₂` also terminates.
Such a relation `tr` is also known as a refinement. -/
def Respects {σ₁ σ₂} (f₁ : σ₁ → Option σ₁) (f₂ : σ₂ → Option σ₂) (tr : σ₁ → σ₂ → Prop) :=
  ∀ ⦃a₁ a₂⦄, tr a₁ a₂ → (match f₁ a₁ with
    | some b₁ => ∃ b₂, tr b₁ b₂ ∧ Reaches₁ f₂ a₂ b₂
    | none => f₂ a₂ = none : Prop)


theorem tr_reaches₁ {σ₁ σ₂ f₁ f₂} {tr : σ₁ → σ₂ → Prop} (H : Respects f₁ f₂ tr) {a₁ a₂}
    (aa : tr a₁ a₂) {b₁} (ab : Reaches₁ f₁ a₁ b₁) : ∃ b₂, tr b₁ b₂ ∧ Reaches₁ f₂ a₂ b₂ := by
  /-
    σ₁ : Type u_1
    σ₂ : Type u_2
    f₁ : σ₁ → Option σ₁
    f₂ : σ₂ → Option σ₂
    tr : σ₁ → σ₂ → Prop
    H : Turing.Respects f₁ f₂ tr
    a₁ : σ₁
    a₂ : σ₂
    aa : tr a₁ a₂
    b₁ : σ₁
    ab : Turing.Reaches₁ f₁ a₁ b₁
    ⊢ Exists fun b₂ => And (tr b₁ b₂) (Turing.Reaches₁ f₂ a₂ b₂)
  -/
  induction' ab with c₁ ac c₁ d₁ _ cd IH
    /-
      case single
      σ₁ : Type u_1
      σ₂ : Type u_2
      f₁ : σ₁ → Option σ₁
      f₂ : σ₂ → Option σ₂
      tr : σ₁ → σ₂ → Prop
      H : Turing.Respects f₁ f₂ tr
      a₁ : σ₁
      a₂ : σ₂
      aa : tr a₁ a₂
      b₁ c₁ : σ₁
      ac : Membership.mem (f₁ a₁) c₁
      ⊢ Exists fun b₂ => And (tr c₁ b₂) (Turing.Reaches₁ f₂ a₂ b₂)
    -/
  · have := H aa
    /-
      case single
      σ₁ : Type u_1
      σ₂ : Type u_2
      f₁ : σ₁ → Option σ₁
      f₂ : σ₂ → Option σ₂
      tr : σ₁ → σ₂ → Prop
      H : Turing.Respects f₁ f₂ tr
      a₁ : σ₁
      a₂ : σ₂
      aa : tr a₁ a₂
      b₁ c₁ : σ₁
      ac : Membership.mem (f₁ a₁) c₁
      this : Turing.Respects.match_1 (fun x => Prop) (f₁ a₁) (fun b₁ => Exists fun b …
      ⊢ Exists fun b₂ => And (tr c₁ b₂) (Turing.Reaches₁ f₂ a₂ b₂)
    -/
    rwa [show f₁ a₁ = _ from ac] at this
    /-
      🎉 no goals
    -/
    /-
      case tail
      σ₁ : Type u_1
      σ₂ : Type u_2
      f₁ : σ₁ → Option σ₁
      f₂ : σ₂ → Option σ₂
      tr : σ₁ → σ₂ → Prop
      H : Turing.Respects f₁ f₂ tr
      a₁ : σ₁
      a₂ : σ₂
      aa : tr a₁ a₂
      b₁ c₁ d₁ : σ₁
      a✝ : Relation.TransGen (fun a b => Membership.mem (f₁ a) b) a₁ c₁
      cd : Membership.mem (f₁ c₁) d₁
      IH : Exists fun b₂ => And (tr c₁ b₂) (Turing.Reaches₁ f₂ a₂ b₂)
      ⊢ Exists fun b₂ => And (tr d₁ b₂) (Turing.Reaches₁ f₂ a₂ b₂)
    -/
  · rcases IH with ⟨c₂, cc, ac₂⟩
    /-
      case tail.intro.intro
      σ₁ : Type u_1
      σ₂ : Type u_2
      f₁ : σ₁ → Option σ₁
      f₂ : σ₂ → Option σ₂
      tr : σ₁ → σ₂ → Prop
      H : Turing.Respects f₁ f₂ tr
      a₁ : σ₁
      a₂ : σ₂
      aa : tr a₁ a₂
      b₁ c₁ d₁ : σ₁
      a✝ : Relation.TransGen (fun a b => Membership.mem (f₁ a) b) a₁ c₁
      cd : Membership.mem (f₁ c₁) d₁
      c₂ : σ₂
      cc : tr c₁ c₂
      ac₂ : Turing.Reaches₁ f₂ a₂ c₂
      ⊢ Exists fun b₂ => And (tr d₁ b₂) (Turing.Reaches₁ f₂ a₂ b₂)
    -/
    have := H cc
    /-
      case tail.intro.intro
      σ₁ : Type u_1
      σ₂ : Type u_2
      f₁ : σ₁ → Option σ₁
      f₂ : σ₂ → Option σ₂
      tr : σ₁ → σ₂ → Prop
      H : Turing.Respects f₁ f₂ tr
      a₁ : σ₁
      a₂ : σ₂
      aa : tr a₁ a₂
      b₁ c₁ d₁ : σ₁
      a✝ : Relation.TransGen (fun a b => Membership.mem (f₁ a) b) a₁ c₁
      cd : Membership.mem (f₁ c₁) d₁
      c₂ : σ₂
      cc : tr c₁ c₂
      ac₂ : Turing.Reaches₁ f₂ a₂ c₂
      this : Turing.Respects.match_1 (fun x => Prop) (f₁ c₁) (fun b₁ => Exists fun b …
      ⊢ Exists fun b₂ => And (tr d₁ b₂) (Turing.Reaches₁ f₂ a₂ b₂)
    -/
    rw [show f₁ c₁ = _ from cd] at this
    /-
      case tail.intro.intro
      σ₁ : Type u_1
      σ₂ : Type u_2
      f₁ : σ₁ → Option σ₁
      f₂ : σ₂ → Option σ₂
      tr : σ₁ → σ₂ → Prop
      H : Turing.Respects f₁ f₂ tr
      a₁ : σ₁
      a₂ : σ₂
      aa : tr a₁ a₂
      b₁ c₁ d₁ : σ₁
      a✝ : Relation.TransGen (fun a b => Membership.mem (f₁ a) b) a₁ c₁
      cd : Membership.mem (f₁ c₁) d₁
      c₂ : σ₂
      cc : tr c₁ c₂
      ac₂ : Turing.Reaches₁ f₂ a₂ c₂
      this : Turing.Respects.match_1 (fun x => Prop) (Option.some d₁) (fun b₁ => Exi …
      ⊢ Exists fun b₂ => And (tr d₁ b₂) (Turing.Reaches₁ f₂ a₂ b₂)
    -/
    rcases this with ⟨d₂, dd, cd₂⟩
    /-
      case tail.intro.intro.intro.intro
      σ₁ : Type u_1
      σ₂ : Type u_2
      f₁ : σ₁ → Option σ₁
      f₂ : σ₂ → Option σ₂
      tr : σ₁ → σ₂ → Prop
      H : Turing.Respects f₁ f₂ tr
      a₁ : σ₁
      a₂ : σ₂
      aa : tr a₁ a₂
      b₁ c₁ d₁ : σ₁
      a✝ : Relation.TransGen (fun a b => Membership.mem (f₁ a) b) a₁ c₁
      cd : Membership.mem (f₁ c₁) d₁
      c₂ : σ₂
      cc : tr c₁ c₂
      ac₂ : Turing.Reaches₁ f₂ a₂ c₂
      d₂ : σ₂
      dd : tr d₁ d₂
      cd₂ : Turing.Reaches₁ f₂ c₂ d₂
      ⊢ Exists fun b₂ => And (tr d₁ b₂) (Turing.Reaches₁ f₂ a₂ b₂)
    -/
    exact ⟨_, dd, ac₂.trans cd₂⟩
    /-
      🎉 no goals
    -/


theorem tr_reaches {σ₁ σ₂ f₁ f₂} {tr : σ₁ → σ₂ → Prop} (H : Respects f₁ f₂ tr) {a₁ a₂}
    (aa : tr a₁ a₂) {b₁} (ab : Reaches f₁ a₁ b₁) : ∃ b₂, tr b₁ b₂ ∧ Reaches f₂ a₂ b₂ := by
  /-
    σ₁ : Type u_1
    σ₂ : Type u_2
    f₁ : σ₁ → Option σ₁
    f₂ : σ₂ → Option σ₂
    tr : σ₁ → σ₂ → Prop
    H : Turing.Respects f₁ f₂ tr
    a₁ : σ₁
    a₂ : σ₂
    aa : tr a₁ a₂
    b₁ : σ₁
    ab : Turing.Reaches f₁ a₁ b₁
    ⊢ Exists fun b₂ => And (tr b₁ b₂) (Turing.Reaches f₂ a₂ b₂)
  -/
  rcases reflTransGen_iff_eq_or_transGen.1 ab with (rfl | ab)
    /-
      case inl
      σ₁ : Type u_1
      σ₂ : Type u_2
      f₁ : σ₁ → Option σ₁
      f₂ : σ₂ → Option σ₂
      tr : σ₁ → σ₂ → Prop
      H : Turing.Respects f₁ f₂ tr
      a₂ : σ₂
      b₁ : σ₁
      aa : tr b₁ a₂
      ab : Turing.Reaches f₁ b₁ b₁
      ⊢ Exists fun b₂ => And (tr b₁ b₂) (Turing.Reaches f₂ a₂ b₂)
    -/
  · exact ⟨_, aa, ReflTransGen.refl⟩
    /-
      🎉 no goals
    -/
    /-
      case inr
      σ₁ : Type u_1
      σ₂ : Type u_2
      f₁ : σ₁ → Option σ₁
      f₂ : σ₂ → Option σ₂
      tr : σ₁ → σ₂ → Prop
      H : Turing.Respects f₁ f₂ tr
      a₁ : σ₁
      a₂ : σ₂
      aa : tr a₁ a₂
      b₁ : σ₁
      ab✝ : Turing.Reaches f₁ a₁ b₁
      ab : Relation.TransGen (fun a b => Membership.mem (f₁ a) b) a₁ b₁
      ⊢ Exists fun b₂ => And (tr b₁ b₂) (Turing.Reaches f₂ a₂ b₂)
    -/
  · have ⟨b₂, bb, h⟩ := tr_reaches₁ H aa ab
    /-
      case inr
      σ₁ : Type u_1
      σ₂ : Type u_2
      f₁ : σ₁ → Option σ₁
      f₂ : σ₂ → Option σ₂
      tr : σ₁ → σ₂ → Prop
      H : Turing.Respects f₁ f₂ tr
      a₁ : σ₁
      a₂ : σ₂
      aa : tr a₁ a₂
      b₁ : σ₁
      ab✝ : Turing.Reaches f₁ a₁ b₁
      ab : Relation.TransGen (fun a b => Membership.mem (f₁ a) b) a₁ b₁
      b₂ : σ₂
      bb : tr b₁ b₂
      h : Turing.Reaches₁ f₂ a₂ b₂
      ⊢ Exists fun b₂ => And (tr b₁ b₂) (Turing.Reaches f₂ a₂ b₂)
    -/
    exact ⟨b₂, bb, h.to_reflTransGen⟩
    /-
      🎉 no goals
    -/


theorem tr_reaches_rev {σ₁ σ₂ f₁ f₂} {tr : σ₁ → σ₂ → Prop} (H : Respects f₁ f₂ tr) {a₁ a₂}
    (aa : tr a₁ a₂) {b₂} (ab : Reaches f₂ a₂ b₂) :
    ∃ c₁ c₂, Reaches f₂ b₂ c₂ ∧ tr c₁ c₂ ∧ Reaches f₁ a₁ c₁ := by
  /-
    σ₁ : Type u_1
    σ₂ : Type u_2
    f₁ : σ₁ → Option σ₁
    f₂ : σ₂ → Option σ₂
    tr : σ₁ → σ₂ → Prop
    H : Turing.Respects f₁ f₂ tr
    a₁ : σ₁
    a₂ : σ₂
    aa : tr a₁ a₂
    b₂ : σ₂
    ab : Turing.Reaches f₂ a₂ b₂
    ⊢ Exists fun c₁ => Exists fun c₂ => And (Turing.Reaches f₂ b₂ c₂) (And (tr c₁  …
  -/
  induction' ab with c₂ d₂ _ cd IH
    /-
      case refl
      σ₁ : Type u_1
      σ₂ : Type u_2
      f₁ : σ₁ → Option σ₁
      f₂ : σ₂ → Option σ₂
      tr : σ₁ → σ₂ → Prop
      H : Turing.Respects f₁ f₂ tr
      a₁ : σ₁
      a₂ : σ₂
      aa : tr a₁ a₂
      b₂ : σ₂
      ⊢ Exists fun c₁ => Exists fun c₂ => And (Turing.Reaches f₂ a₂ c₂) (And (tr c₁  …
    -/
  · exact ⟨_, _, ReflTransGen.refl, aa, ReflTransGen.refl⟩
    /-
      🎉 no goals
    -/
    /-
      case tail
      σ₁ : Type u_1
      σ₂ : Type u_2
      f₁ : σ₁ → Option σ₁
      f₂ : σ₂ → Option σ₂
      tr : σ₁ → σ₂ → Prop
      H : Turing.Respects f₁ f₂ tr
      a₁ : σ₁
      a₂ : σ₂
      aa : tr a₁ a₂
      b₂ c₂ d₂ : σ₂
      a✝ : Relation.ReflTransGen (fun a b => Membership.mem (f₂ a) b) a₂ c₂
      cd : Membership.mem (f₂ c₂) d₂
      IH : Exists fun c₁ => Exists fun c₂_1 => And (Turing.Reaches f₂ c₂ c₂_1) (And  …
      ⊢ Exists fun c₁ => Exists fun c₂ => And (Turing.Reaches f₂ d₂ c₂) (And (tr c₁  …
    -/
  · rcases IH with ⟨e₁, e₂, ce, ee, ae⟩
    /-
      case tail.intro.intro.intro.intro
      σ₁ : Type u_1
      σ₂ : Type u_2
      f₁ : σ₁ → Option σ₁
      f₂ : σ₂ → Option σ₂
      tr : σ₁ → σ₂ → Prop
      H : Turing.Respects f₁ f₂ tr
      a₁ : σ₁
      a₂ : σ₂
      aa : tr a₁ a₂
      b₂ c₂ d₂ : σ₂
      a✝ : Relation.ReflTransGen (fun a b => Membership.mem (f₂ a) b) a₂ c₂
      cd : Membership.mem (f₂ c₂) d₂
      e₁ : σ₁
      e₂ : σ₂
      ce : Turing.Reaches f₂ c₂ e₂
      ee : tr e₁ e₂
      ae : Turing.Reaches f₁ a₁ e₁
      ⊢ Exists fun c₁ => Exists fun c₂ => And (Turing.Reaches f₂ d₂ c₂) (And (tr c₁  …
    -/
    rcases ReflTransGen.cases_head ce with (rfl | ⟨d', cd', de⟩)
      /-
        case tail.intro.intro.intro.intro.inl
        σ₁ : Type u_1
        σ₂ : Type u_2
        f₁ : σ₁ → Option σ₁
        f₂ : σ₂ → Option σ₂
        tr : σ₁ → σ₂ → Prop
        H : Turing.Respects f₁ f₂ tr
        a₁ : σ₁
        a₂ : σ₂
        aa : tr a₁ a₂
        b₂ c₂ d₂ : σ₂
        a✝ : Relation.ReflTransGen (fun a b => Membership.mem (f₂ a) b) a₂ c₂
        cd : Membership.mem (f₂ c₂) d₂
        e₁ : σ₁
        ae : Turing.Reaches f₁ a₁ e₁
        ce : Turing.Reaches f₂ c₂ c₂
        ee : tr e₁ c₂
        ⊢ Exists fun c₁ => Exists fun c₂ => And (Turing.Reaches f₂ d₂ c₂) (And (tr c₁  …
      -/
    · have := H ee
      /-
        case tail.intro.intro.intro.intro.inl
        σ₁ : Type u_1
        σ₂ : Type u_2
        f₁ : σ₁ → Option σ₁
        f₂ : σ₂ → Option σ₂
        tr : σ₁ → σ₂ → Prop
        H : Turing.Respects f₁ f₂ tr
        a₁ : σ₁
        a₂ : σ₂
        aa : tr a₁ a₂
        b₂ c₂ d₂ : σ₂
        a✝ : Relation.ReflTransGen (fun a b => Membership.mem (f₂ a) b) a₂ c₂
        cd : Membership.mem (f₂ c₂) d₂
        e₁ : σ₁
        ae : Turing.Reaches f₁ a₁ e₁
        ce : Turing.Reaches f₂ c₂ c₂
        ee : tr e₁ c₂
        this : Turing.Respects.match_1 (fun x => Prop) (f₁ e₁) (fun b₁ => Exists fun b …
        ⊢ Exists fun c₁ => Exists fun c₂ => And (Turing.Reaches f₂ d₂ c₂) (And (tr c₁  …
      -/
      revert this
      /-
        case tail.intro.intro.intro.intro.inl
        σ₁ : Type u_1
        σ₂ : Type u_2
        f₁ : σ₁ → Option σ₁
        f₂ : σ₂ → Option σ₂
        tr : σ₁ → σ₂ → Prop
        H : Turing.Respects f₁ f₂ tr
        a₁ : σ₁
        a₂ : σ₂
        aa : tr a₁ a₂
        b₂ c₂ d₂ : σ₂
        a✝ : Relation.ReflTransGen (fun a b => Membership.mem (f₂ a) b) a₂ c₂
        cd : Membership.mem (f₂ c₂) d₂
        e₁ : σ₁
        ae : Turing.Reaches f₁ a₁ e₁
        ce : Turing.Reaches f₂ c₂ c₂
        ee : tr e₁ c₂
        ⊢ (Turing.Respects.match_1 (fun x => Prop) (f₁ e₁) (fun b₁ => Exists fun b₂ => …
      -/
      cases' eg : f₁ e₁ with g₁ <;> simp only [Respects, and_imp, exists_imp]
        /-
          case tail.intro.intro.intro.intro.inl.none
          σ₁ : Type u_1
          σ₂ : Type u_2
          f₁ : σ₁ → Option σ₁
          f₂ : σ₂ → Option σ₂
          tr : σ₁ → σ₂ → Prop
          H : Turing.Respects f₁ f₂ tr
          a₁ : σ₁
          a₂ : σ₂
          aa : tr a₁ a₂
          b₂ c₂ d₂ : σ₂
          a✝ : Relation.ReflTransGen (fun a b => Membership.mem (f₂ a) b) a₂ c₂
          cd : Membership.mem (f₂ c₂) d₂
          e₁ : σ₁
          ae : Turing.Reaches f₁ a₁ e₁
          ce : Turing.Reaches f₂ c₂ c₂
          ee : tr e₁ c₂
          eg : Eq (f₁ e₁) Option.none
          ⊢ Eq (f₂ c₂) Option.none → Exists fun c₁ => Exists fun c₂ => And (Turing.Reach …
        -/
      · intro c0
        /-
          case tail.intro.intro.intro.intro.inl.none
          σ₁ : Type u_1
          σ₂ : Type u_2
          f₁ : σ₁ → Option σ₁
          f₂ : σ₂ → Option σ₂
          tr : σ₁ → σ₂ → Prop
          H : Turing.Respects f₁ f₂ tr
          a₁ : σ₁
          a₂ : σ₂
          aa : tr a₁ a₂
          b₂ c₂ d₂ : σ₂
          a✝ : Relation.ReflTransGen (fun a b => Membership.mem (f₂ a) b) a₂ c₂
          cd : Membership.mem (f₂ c₂) d₂
          e₁ : σ₁
          ae : Turing.Reaches f₁ a₁ e₁
          ce : Turing.Reaches f₂ c₂ c₂
          ee : tr e₁ c₂
          eg : Eq (f₁ e₁) Option.none
          c0 : Eq (f₂ c₂) Option.none
          ⊢ Exists fun c₁ => Exists fun c₂ => And (Turing.Reaches f₂ d₂ c₂) (And (tr c₁  …
        -/
        cases cd.symm.trans c0
        /-
          🎉 no goals
        -/
        /-
          case tail.intro.intro.intro.intro.inl.some
          σ₁ : Type u_1
          σ₂ : Type u_2
          f₁ : σ₁ → Option σ₁
          f₂ : σ₂ → Option σ₂
          tr : σ₁ → σ₂ → Prop
          H : Turing.Respects f₁ f₂ tr
          a₁ : σ₁
          a₂ : σ₂
          aa : tr a₁ a₂
          b₂ c₂ d₂ : σ₂
          a✝ : Relation.ReflTransGen (fun a b => Membership.mem (f₂ a) b) a₂ c₂
          cd : Membership.mem (f₂ c₂) d₂
          e₁ : σ₁
          ae : Turing.Reaches f₁ a₁ e₁
          ce : Turing.Reaches f₂ c₂ c₂
          ee : tr e₁ c₂
          g₁ : σ₁
          eg : Eq (f₁ e₁) (Option.some g₁)
          ⊢ ∀ (x : σ₂), tr g₁ x → Turing.Reaches₁ f₂ c₂ x → Exists fun c₁ => Exists fun  …
        -/
      · intro g₂ gg cg
        /-
          case tail.intro.intro.intro.intro.inl.some
          σ₁ : Type u_1
          σ₂ : Type u_2
          f₁ : σ₁ → Option σ₁
          f₂ : σ₂ → Option σ₂
          tr : σ₁ → σ₂ → Prop
          H : Turing.Respects f₁ f₂ tr
          a₁ : σ₁
          a₂ : σ₂
          aa : tr a₁ a₂
          b₂ c₂ d₂ : σ₂
          a✝ : Relation.ReflTransGen (fun a b => Membership.mem (f₂ a) b) a₂ c₂
          cd : Membership.mem (f₂ c₂) d₂
          e₁ : σ₁
          ae : Turing.Reaches f₁ a₁ e₁
          ce : Turing.Reaches f₂ c₂ c₂
          ee : tr e₁ c₂
          g₁ : σ₁
          eg : Eq (f₁ e₁) (Option.some g₁)
          g₂ : σ₂
          gg : tr g₁ g₂
          cg : Turing.Reaches₁ f₂ c₂ g₂
          ⊢ Exists fun c₁ => Exists fun c₂ => And (Turing.Reaches f₂ d₂ c₂) (And (tr c₁  …
        -/
        rcases TransGen.head'_iff.1 cg with ⟨d', cd', dg⟩
        /-
          case tail.intro.intro.intro.intro.inl.some.intro.intro
          σ₁ : Type u_1
          σ₂ : Type u_2
          f₁ : σ₁ → Option σ₁
          f₂ : σ₂ → Option σ₂
          tr : σ₁ → σ₂ → Prop
          H : Turing.Respects f₁ f₂ tr
          a₁ : σ₁
          a₂ : σ₂
          aa : tr a₁ a₂
          b₂ c₂ d₂ : σ₂
          a✝ : Relation.ReflTransGen (fun a b => Membership.mem (f₂ a) b) a₂ c₂
          cd : Membership.mem (f₂ c₂) d₂
          e₁ : σ₁
          ae : Turing.Reaches f₁ a₁ e₁
          ce : Turing.Reaches f₂ c₂ c₂
          ee : tr e₁ c₂
          g₁ : σ₁
          eg : Eq (f₁ e₁) (Option.some g₁)
          g₂ : σ₂
          gg : tr g₁ g₂
          cg : Turing.Reaches₁ f₂ c₂ g₂
          d' : σ₂
          cd' : Membership.mem (f₂ c₂) d'
          dg : Relation.ReflTransGen (fun a b => Membership.mem (f₂ a) b) d' g₂
          ⊢ Exists fun c₁ => Exists fun c₂ => And (Turing.Reaches f₂ d₂ c₂) (And (tr c₁  …
        -/
        cases Option.mem_unique cd cd'
        /-
          case tail.intro.intro.intro.intro.inl.some.intro.intro.refl
          σ₁ : Type u_1
          σ₂ : Type u_2
          f₁ : σ₁ → Option σ₁
          f₂ : σ₂ → Option σ₂
          tr : σ₁ → σ₂ → Prop
          H : Turing.Respects f₁ f₂ tr
          a₁ : σ₁
          a₂ : σ₂
          aa : tr a₁ a₂
          b₂ c₂ d₂ : σ₂
          a✝ : Relation.ReflTransGen (fun a b => Membership.mem (f₂ a) b) a₂ c₂
          cd : Membership.mem (f₂ c₂) d₂
          e₁ : σ₁
          ae : Turing.Reaches f₁ a₁ e₁
          ce : Turing.Reaches f₂ c₂ c₂
          ee : tr e₁ c₂
          g₁ : σ₁
          eg : Eq (f₁ e₁) (Option.some g₁)
          g₂ : σ₂
          gg : tr g₁ g₂
          cg : Turing.Reaches₁ f₂ c₂ g₂
          cd' : Membership.mem (f₂ c₂) d₂
          dg : Relation.ReflTransGen (fun a b => Membership.mem (f₂ a) b) d₂ g₂
          ⊢ Exists fun c₁ => Exists fun c₂ => And (Turing.Reaches f₂ d₂ c₂) (And (tr c₁  …
        -/
        exact ⟨_, _, dg, gg, ae.tail eg⟩
        /-
          🎉 no goals
        -/
      /-
        case tail.intro.intro.intro.intro.inr.intro.intro
        σ₁ : Type u_1
        σ₂ : Type u_2
        f₁ : σ₁ → Option σ₁
        f₂ : σ₂ → Option σ₂
        tr : σ₁ → σ₂ → Prop
        H : Turing.Respects f₁ f₂ tr
        a₁ : σ₁
        a₂ : σ₂
        aa : tr a₁ a₂
        b₂ c₂ d₂ : σ₂
        a✝ : Relation.ReflTransGen (fun a b => Membership.mem (f₂ a) b) a₂ c₂
        cd : Membership.mem (f₂ c₂) d₂
        e₁ : σ₁
        e₂ : σ₂
        ce : Turing.Reaches f₂ c₂ e₂
        ee : tr e₁ e₂
        ae : Turing.Reaches f₁ a₁ e₁
        d' : σ₂
        cd' : Membership.mem (f₂ c₂) d'
        de : Relation.ReflTransGen (fun a b => Membership.mem (f₂ a) b) d' e₂
        ⊢ Exists fun c₁ => Exists fun c₂ => And (Turing.Reaches f₂ d₂ c₂) (And (tr c₁  …
      -/
    · cases Option.mem_unique cd cd'
      /-
        case tail.intro.intro.intro.intro.inr.intro.intro.refl
        σ₁ : Type u_1
        σ₂ : Type u_2
        f₁ : σ₁ → Option σ₁
        f₂ : σ₂ → Option σ₂
        tr : σ₁ → σ₂ → Prop
        H : Turing.Respects f₁ f₂ tr
        a₁ : σ₁
        a₂ : σ₂
        aa : tr a₁ a₂
        b₂ c₂ d₂ : σ₂
        a✝ : Relation.ReflTransGen (fun a b => Membership.mem (f₂ a) b) a₂ c₂
        cd : Membership.mem (f₂ c₂) d₂
        e₁ : σ₁
        e₂ : σ₂
        ce : Turing.Reaches f₂ c₂ e₂
        ee : tr e₁ e₂
        ae : Turing.Reaches f₁ a₁ e₁
        cd' : Membership.mem (f₂ c₂) d₂
        de : Relation.ReflTransGen (fun a b => Membership.mem (f₂ a) b) d₂ e₂
        ⊢ Exists fun c₁ => Exists fun c₂ => And (Turing.Reaches f₂ d₂ c₂) (And (tr c₁  …
      -/
      exact ⟨_, _, de, ee, ae⟩
      /-
        🎉 no goals
      -/


theorem tr_eval {σ₁ σ₂ f₁ f₂} {tr : σ₁ → σ₂ → Prop} (H : Respects f₁ f₂ tr) {a₁ b₁ a₂}
    (aa : tr a₁ a₂) (ab : b₁ ∈ eval f₁ a₁) : ∃ b₂, tr b₁ b₂ ∧ b₂ ∈ eval f₂ a₂ := by
  /-
    σ₁ : Type u_1
    σ₂ : Type u_2
    f₁ : σ₁ → Option σ₁
    f₂ : σ₂ → Option σ₂
    tr : σ₁ → σ₂ → Prop
    H : Turing.Respects f₁ f₂ tr
    a₁ b₁ : σ₁
    a₂ : σ₂
    aa : tr a₁ a₂
    ab : Membership.mem (Turing.eval f₁ a₁) b₁
    ⊢ Exists fun b₂ => And (tr b₁ b₂) (Membership.mem (Turing.eval f₂ a₂) b₂)
  -/
  cases' mem_eval.1 ab with ab b0
  /-
    case intro
    σ₁ : Type u_1
    σ₂ : Type u_2
    f₁ : σ₁ → Option σ₁
    f₂ : σ₂ → Option σ₂
    tr : σ₁ → σ₂ → Prop
    H : Turing.Respects f₁ f₂ tr
    a₁ b₁ : σ₁
    a₂ : σ₂
    aa : tr a₁ a₂
    ab✝ : Membership.mem (Turing.eval f₁ a₁) b₁
    ab : Turing.Reaches f₁ a₁ b₁
    b0 : Eq (f₁ b₁) Option.none
    ⊢ Exists fun b₂ => And (tr b₁ b₂) (Membership.mem (Turing.eval f₂ a₂) b₂)
  -/
  rcases tr_reaches H aa ab with ⟨b₂, bb, ab⟩
  /-
    case intro.intro.intro
    σ₁ : Type u_1
    σ₂ : Type u_2
    f₁ : σ₁ → Option σ₁
    f₂ : σ₂ → Option σ₂
    tr : σ₁ → σ₂ → Prop
    H : Turing.Respects f₁ f₂ tr
    a₁ b₁ : σ₁
    a₂ : σ₂
    aa : tr a₁ a₂
    ab✝¹ : Membership.mem (Turing.eval f₁ a₁) b₁
    ab✝ : Turing.Reaches f₁ a₁ b₁
    b0 : Eq (f₁ b₁) Option.none
    b₂ : σ₂
    bb : tr b₁ b₂
    ab : Turing.Reaches f₂ a₂ b₂
    ⊢ Exists fun b₂ => And (tr b₁ b₂) (Membership.mem (Turing.eval f₂ a₂) b₂)
  -/
  refine ⟨_, bb, mem_eval.2 ⟨ab, ?_⟩⟩
  /-
    case intro.intro.intro
    σ₁ : Type u_1
    σ₂ : Type u_2
    f₁ : σ₁ → Option σ₁
    f₂ : σ₂ → Option σ₂
    tr : σ₁ → σ₂ → Prop
    H : Turing.Respects f₁ f₂ tr
    a₁ b₁ : σ₁
    a₂ : σ₂
    aa : tr a₁ a₂
    ab✝¹ : Membership.mem (Turing.eval f₁ a₁) b₁
    ab✝ : Turing.Reaches f₁ a₁ b₁
    b0 : Eq (f₁ b₁) Option.none
    b₂ : σ₂
    bb : tr b₁ b₂
    ab : Turing.Reaches f₂ a₂ b₂
    ⊢ Eq (f₂ b₂) Option.none
  -/
  have := H bb; rwa [b0] at this
                /-
                  🎉 no goals
                -/


theorem tr_eval_rev {σ₁ σ₂ f₁ f₂} {tr : σ₁ → σ₂ → Prop} (H : Respects f₁ f₂ tr) {a₁ b₂ a₂}
    (aa : tr a₁ a₂) (ab : b₂ ∈ eval f₂ a₂) : ∃ b₁, tr b₁ b₂ ∧ b₁ ∈ eval f₁ a₁ := by
  /-
    σ₁ : Type u_1
    σ₂ : Type u_2
    f₁ : σ₁ → Option σ₁
    f₂ : σ₂ → Option σ₂
    tr : σ₁ → σ₂ → Prop
    H : Turing.Respects f₁ f₂ tr
    a₁ : σ₁
    b₂ a₂ : σ₂
    aa : tr a₁ a₂
    ab : Membership.mem (Turing.eval f₂ a₂) b₂
    ⊢ Exists fun b₁ => And (tr b₁ b₂) (Membership.mem (Turing.eval f₁ a₁) b₁)
  -/
  cases' mem_eval.1 ab with ab b0
  /-
    case intro
    σ₁ : Type u_1
    σ₂ : Type u_2
    f₁ : σ₁ → Option σ₁
    f₂ : σ₂ → Option σ₂
    tr : σ₁ → σ₂ → Prop
    H : Turing.Respects f₁ f₂ tr
    a₁ : σ₁
    b₂ a₂ : σ₂
    aa : tr a₁ a₂
    ab✝ : Membership.mem (Turing.eval f₂ a₂) b₂
    ab : Turing.Reaches f₂ a₂ b₂
    b0 : Eq (f₂ b₂) Option.none
    ⊢ Exists fun b₁ => And (tr b₁ b₂) (Membership.mem (Turing.eval f₁ a₁) b₁)
  -/
  rcases tr_reaches_rev H aa ab with ⟨c₁, c₂, bc, cc, ac⟩
  /-
    case intro.intro.intro.intro.intro
    σ₁ : Type u_1
    σ₂ : Type u_2
    f₁ : σ₁ → Option σ₁
    f₂ : σ₂ → Option σ₂
    tr : σ₁ → σ₂ → Prop
    H : Turing.Respects f₁ f₂ tr
    a₁ : σ₁
    b₂ a₂ : σ₂
    aa : tr a₁ a₂
    ab✝ : Membership.mem (Turing.eval f₂ a₂) b₂
    ab : Turing.Reaches f₂ a₂ b₂
    b0 : Eq (f₂ b₂) Option.none
    c₁ : σ₁
    c₂ : σ₂
    bc : Turing.Reaches f₂ b₂ c₂
    cc : tr c₁ c₂
    ac : Turing.Reaches f₁ a₁ c₁
    ⊢ Exists fun b₁ => And (tr b₁ b₂) (Membership.mem (Turing.eval f₁ a₁) b₁)
  -/
  cases (reflTransGen_iff_eq (Option.eq_none_iff_forall_not_mem.1 b0)).1 bc
  /-
    case intro.intro.intro.intro.intro.refl
    σ₁ : Type u_1
    σ₂ : Type u_2
    f₁ : σ₁ → Option σ₁
    f₂ : σ₂ → Option σ₂
    tr : σ₁ → σ₂ → Prop
    H : Turing.Respects f₁ f₂ tr
    a₁ : σ₁
    b₂ a₂ : σ₂
    aa : tr a₁ a₂
    ab✝ : Membership.mem (Turing.eval f₂ a₂) b₂
    ab : Turing.Reaches f₂ a₂ b₂
    b0 : Eq (f₂ b₂) Option.none
    c₁ : σ₁
    ac : Turing.Reaches f₁ a₁ c₁
    bc : Turing.Reaches f₂ b₂ b₂
    cc : tr c₁ b₂
    ⊢ Exists fun b₁ => And (tr b₁ b₂) (Membership.mem (Turing.eval f₁ a₁) b₁)
  -/
  refine ⟨_, cc, mem_eval.2 ⟨ac, ?_⟩⟩
  /-
    case intro.intro.intro.intro.intro.refl
    σ₁ : Type u_1
    σ₂ : Type u_2
    f₁ : σ₁ → Option σ₁
    f₂ : σ₂ → Option σ₂
    tr : σ₁ → σ₂ → Prop
    H : Turing.Respects f₁ f₂ tr
    a₁ : σ₁
    b₂ a₂ : σ₂
    aa : tr a₁ a₂
    ab✝ : Membership.mem (Turing.eval f₂ a₂) b₂
    ab : Turing.Reaches f₂ a₂ b₂
    b0 : Eq (f₂ b₂) Option.none
    c₁ : σ₁
    ac : Turing.Reaches f₁ a₁ c₁
    bc : Turing.Reaches f₂ b₂ b₂
    cc : tr c₁ b₂
    ⊢ Eq (f₁ c₁) Option.none
  -/
  have := H cc
  /-
    case intro.intro.intro.intro.intro.refl
    σ₁ : Type u_1
    σ₂ : Type u_2
    f₁ : σ₁ → Option σ₁
    f₂ : σ₂ → Option σ₂
    tr : σ₁ → σ₂ → Prop
    H : Turing.Respects f₁ f₂ tr
    a₁ : σ₁
    b₂ a₂ : σ₂
    aa : tr a₁ a₂
    ab✝ : Membership.mem (Turing.eval f₂ a₂) b₂
    ab : Turing.Reaches f₂ a₂ b₂
    b0 : Eq (f₂ b₂) Option.none
    c₁ : σ₁
    ac : Turing.Reaches f₁ a₁ c₁
    bc : Turing.Reaches f₂ b₂ b₂
    cc : tr c₁ b₂
    this : Turing.Respects.match_1 (fun x => Prop) (f₁ c₁) (fun b₁ => Exists fun b …
    ⊢ Eq (f₁ c₁) Option.none
  -/
  cases' hfc : f₁ c₁ with d₁
    /-
      case intro.intro.intro.intro.intro.refl.none
      σ₁ : Type u_1
      σ₂ : Type u_2
      f₁ : σ₁ → Option σ₁
      f₂ : σ₂ → Option σ₂
      tr : σ₁ → σ₂ → Prop
      H : Turing.Respects f₁ f₂ tr
      a₁ : σ₁
      b₂ a₂ : σ₂
      aa : tr a₁ a₂
      ab✝ : Membership.mem (Turing.eval f₂ a₂) b₂
      ab : Turing.Reaches f₂ a₂ b₂
      b0 : Eq (f₂ b₂) Option.none
      c₁ : σ₁
      ac : Turing.Reaches f₁ a₁ c₁
      bc : Turing.Reaches f₂ b₂ b₂
      cc : tr c₁ b₂
      this : Turing.Respects.match_1 (fun x => Prop) (f₁ c₁) (fun b₁ => Exists fun b …
      hfc : Eq (f₁ c₁) Option.none
      ⊢ Eq Option.none Option.none
    -/
  · rfl
    /-
      🎉 no goals
    -/
  /-
    case intro.intro.intro.intro.intro.refl.some
    σ₁ : Type u_1
    σ₂ : Type u_2
    f₁ : σ₁ → Option σ₁
    f₂ : σ₂ → Option σ₂
    tr : σ₁ → σ₂ → Prop
    H : Turing.Respects f₁ f₂ tr
    a₁ : σ₁
    b₂ a₂ : σ₂
    aa : tr a₁ a₂
    ab✝ : Membership.mem (Turing.eval f₂ a₂) b₂
    ab : Turing.Reaches f₂ a₂ b₂
    b0 : Eq (f₂ b₂) Option.none
    c₁ : σ₁
    ac : Turing.Reaches f₁ a₁ c₁
    bc : Turing.Reaches f₂ b₂ b₂
    cc : tr c₁ b₂
    this : Turing.Respects.match_1 (fun x => Prop) (f₁ c₁) (fun b₁ => Exists fun b …
    d₁ : σ₁
    hfc : Eq (f₁ c₁) (Option.some d₁)
    ⊢ Eq (Option.some d₁) Option.none
  -/
  rw [hfc] at this
  /-
    case intro.intro.intro.intro.intro.refl.some
    σ₁ : Type u_1
    σ₂ : Type u_2
    f₁ : σ₁ → Option σ₁
    f₂ : σ₂ → Option σ₂
    tr : σ₁ → σ₂ → Prop
    H : Turing.Respects f₁ f₂ tr
    a₁ : σ₁
    b₂ a₂ : σ₂
    aa : tr a₁ a₂
    ab✝ : Membership.mem (Turing.eval f₂ a₂) b₂
    ab : Turing.Reaches f₂ a₂ b₂
    b0 : Eq (f₂ b₂) Option.none
    c₁ : σ₁
    ac : Turing.Reaches f₁ a₁ c₁
    bc : Turing.Reaches f₂ b₂ b₂
    cc : tr c₁ b₂
    d₁ : σ₁
    this : Turing.Respects.match_1 (fun x => Prop) (Option.some d₁) (fun b₁ => Exi …
    hfc : Eq (f₁ c₁) (Option.some d₁)
    ⊢ Eq (Option.some d₁) Option.none
  -/
  rcases this with ⟨d₂, _, bd⟩
  /-
    case intro.intro.intro.intro.intro.refl.some.intro.intro
    σ₁ : Type u_1
    σ₂ : Type u_2
    f₁ : σ₁ → Option σ₁
    f₂ : σ₂ → Option σ₂
    tr : σ₁ → σ₂ → Prop
    H : Turing.Respects f₁ f₂ tr
    a₁ : σ₁
    b₂ a₂ : σ₂
    aa : tr a₁ a₂
    ab✝ : Membership.mem (Turing.eval f₂ a₂) b₂
    ab : Turing.Reaches f₂ a₂ b₂
    b0 : Eq (f₂ b₂) Option.none
    c₁ : σ₁
    ac : Turing.Reaches f₁ a₁ c₁
    bc : Turing.Reaches f₂ b₂ b₂
    cc : tr c₁ b₂
    d₁ : σ₁
    hfc : Eq (f₁ c₁) (Option.some d₁)
    d₂ : σ₂
    left✝ : tr d₁ d₂
    bd : Turing.Reaches₁ f₂ b₂ d₂
    ⊢ Eq (Option.some d₁) Option.none
  -/
  rcases TransGen.head'_iff.1 bd with ⟨e, h, _⟩
  /-
    case intro.intro.intro.intro.intro.refl.some.intro.intro.intro.intro
    σ₁ : Type u_1
    σ₂ : Type u_2
    f₁ : σ₁ → Option σ₁
    f₂ : σ₂ → Option σ₂
    tr : σ₁ → σ₂ → Prop
    H : Turing.Respects f₁ f₂ tr
    a₁ : σ₁
    b₂ a₂ : σ₂
    aa : tr a₁ a₂
    ab✝ : Membership.mem (Turing.eval f₂ a₂) b₂
    ab : Turing.Reaches f₂ a₂ b₂
    b0 : Eq (f₂ b₂) Option.none
    c₁ : σ₁
    ac : Turing.Reaches f₁ a₁ c₁
    bc : Turing.Reaches f₂ b₂ b₂
    cc : tr c₁ b₂
    d₁ : σ₁
    hfc : Eq (f₁ c₁) (Option.some d₁)
    d₂ : σ₂
    left✝ : tr d₁ d₂
    bd : Turing.Reaches₁ f₂ b₂ d₂
    e : σ₂
    h : Membership.mem (f₂ b₂) e
    right✝ : Relation.ReflTransGen (fun a b => Membership.mem (f₂ a) b) e d₂
    ⊢ Eq (Option.some d₁) Option.none
  -/
  cases b0.symm.trans h
  /-
    🎉 no goals
  -/


theorem tr_eval_dom {σ₁ σ₂ f₁ f₂} {tr : σ₁ → σ₂ → Prop} (H : Respects f₁ f₂ tr) {a₁ a₂}
    (aa : tr a₁ a₂) : (eval f₂ a₂).Dom ↔ (eval f₁ a₁).Dom :=
  ⟨fun h ↦
    let ⟨_, _, h, _⟩ := tr_eval_rev H aa ⟨h, rfl⟩
    h,
    fun h ↦
    let ⟨_, _, h, _⟩ := tr_eval H aa ⟨h, rfl⟩
    h⟩


/-- A simpler version of `Respects` when the state transition relation `tr` is a function. -/
def FRespects {σ₁ σ₂} (f₂ : σ₂ → Option σ₂) (tr : σ₁ → σ₂) (a₂ : σ₂) : Option σ₁ → Prop
  | some b₁ => Reaches₁ f₂ a₂ (tr b₁)
  | none => f₂ a₂ = none


theorem frespects_eq {σ₁ σ₂} {f₂ : σ₂ → Option σ₂} {tr : σ₁ → σ₂} {a₂ b₂} (h : f₂ a₂ = f₂ b₂) :
    ∀ {b₁}, FRespects f₂ tr a₂ b₁ ↔ FRespects f₂ tr b₂ b₁
  | some _ => reaches₁_eq h
               /-
                 σ₁ : Type u_1
                 σ₂ : Type u_2
                 f₂ : σ₂ → Option σ₂
                 tr : σ₁ → σ₂
                 a₂ b₂ : σ₂
                 h : Eq (f₂ a₂) (f₂ b₂)
                 ⊢ Iff (Turing.FRespects f₂ tr a₂ Option.none) (Turing.FRespects f₂ tr b₂ Optio …
               -/
  | none => by unfold FRespects; rw [h]
                                 /-
                                   🎉 no goals
                                 -/


theorem fun_respects {σ₁ σ₂ f₁ f₂} {tr : σ₁ → σ₂} :
    (Respects f₁ f₂ fun a b ↦ tr a = b) ↔ ∀ ⦃a₁⦄, FRespects f₂ tr (tr a₁) (f₁ a₁) :=
  forall_congr' fun a₁ ↦ by
    /-
      σ₁ : Type u_1
      σ₂ : Type u_2
      f₁ : σ₁ → Option σ₁
      f₂ : σ₂ → Option σ₂
      tr : σ₁ → σ₂
      a₁ : σ₁
      ⊢ Iff (∀ ⦃a₂ : σ₂⦄, (fun a b => Eq (tr a) b) a₁ a₂ → Turing.Respects.match_1 ( …
    -/
                    /-
                      🎉 no goals
                    -/
    cases f₁ a₁ <;> simp only [FRespects, Respects, exists_eq_left', forall_eq']
                    /-
                      🎉 no goals
                    -/


theorem tr_eval' {σ₁ σ₂} (f₁ : σ₁ → Option σ₁) (f₂ : σ₂ → Option σ₂) (tr : σ₁ → σ₂)
    (H : Respects f₁ f₂ fun a b ↦ tr a = b) (a₁) : eval f₂ (tr a₁) = tr <$> eval f₁ a₁ :=
  Part.ext fun b₂ ↦
    ⟨fun h ↦
      let ⟨b₁, bb, hb⟩ := tr_eval_rev H rfl h
      (Part.mem_map_iff _).2 ⟨b₁, hb, bb⟩,
      fun h ↦ by
      /-
        σ₁ σ₂ : Type u_1
        f₁ : σ₁ → Option σ₁
        f₂ : σ₂ → Option σ₂
        tr : σ₁ → σ₂
        H : Turing.Respects f₁ f₂ fun a b => Eq (tr a) b
        a₁ : σ₁
        b₂ : σ₂
        h : Membership.mem (Functor.map tr (Turing.eval f₁ a₁)) b₂
        ⊢ Membership.mem (Turing.eval f₂ (tr a₁)) b₂
      -/
      rcases (Part.mem_map_iff _).1 h with ⟨b₁, ab, bb⟩
      /-
        case intro.intro
        σ₁ σ₂ : Type u_1
        f₁ : σ₁ → Option σ₁
        f₂ : σ₂ → Option σ₂
        tr : σ₁ → σ₂
        H : Turing.Respects f₁ f₂ fun a b => Eq (tr a) b
        a₁ : σ₁
        b₂ : σ₂
        h : Membership.mem (Functor.map tr (Turing.eval f₁ a₁)) b₂
        b₁ : σ₁
        ab : Membership.mem (Turing.eval f₁ a₁) b₁
        bb : Eq (tr b₁) b₂
        ⊢ Membership.mem (Turing.eval f₂ (tr a₁)) b₂
      -/
      rcases tr_eval H rfl ab with ⟨_, rfl, h⟩
      /-
        case intro.intro.intro.intro
        σ₁ σ₂ : Type u_1
        f₁ : σ₁ → Option σ₁
        f₂ : σ₂ → Option σ₂
        tr : σ₁ → σ₂
        H : Turing.Respects f₁ f₂ fun a b => Eq (tr a) b
        a₁ : σ₁
        b₂ : σ₂
        h✝ : Membership.mem (Functor.map tr (Turing.eval f₁ a₁)) b₂
        b₁ : σ₁
        ab : Membership.mem (Turing.eval f₁ a₁) b₁
        bb : Eq (tr b₁) b₂
        h : Membership.mem (Turing.eval f₂ (tr a₁)) (tr b₁)
        ⊢ Membership.mem (Turing.eval f₂ (tr a₁)) b₂
      -/
      rwa [bb] at h⟩
      /-
        🎉 no goals
      -/


/-- A Turing machine "statement" is just a command to either move
  left or right, or write a symbol on the tape. -/
inductive Stmt
  | move : Dir → Stmt
  | write : Γ → Stmt


local notation "Stmt₀" => Stmt Γ  -- Porting note (https://github.com/leanprover-community/mathlib4/issues/10750): added this to clean up types.


instance Stmt.inhabited [Inhabited Γ] : Inhabited Stmt₀ :=
  ⟨Stmt.write default⟩


/-- A Post-Turing machine with symbol type `Γ` and label type `Λ`
  is a function which, given the current state `q : Λ` and
  the tape head `a : Γ`, either halts (returns `none`) or returns
  a new state `q' : Λ` and a `Stmt` describing what to do,
  either a move left or right, or a write command.

  Both `Λ` and `Γ` are required to be inhabited; the default value
  for `Γ` is the "blank" tape value, and the default value of `Λ` is
  the initial state. -/
@[nolint unusedArguments] -- this is a deliberate addition, see comment
def Machine [Inhabited Λ] :=
  Λ → Γ → Option (Λ × Stmt₀)


local notation "Machine₀" => Machine Γ Λ  -- Porting note (https://github.com/leanprover-community/mathlib4/issues/10750): added this to clean up types.


instance Machine.inhabited [Inhabited Λ] : Inhabited Machine₀ := by
  /-
    Γ : Type u_1
    Λ : Type u_2
    inst✝ : Inhabited Λ
    ⊢ Inhabited (Turing.TM0.Machine Γ Λ)
  -/
  unfold Machine; infer_instance
                  /-
                    🎉 no goals
                  -/


/-- The configuration state of a Turing machine during operation
  consists of a label (machine state), and a tape.
  The tape is represented in the form `(a, L, R)`, meaning the tape looks like `L.rev ++ [a] ++ R`
  with the machine currently reading the `a`. The lists are
  automatically extended with blanks as the machine moves around. -/
structure Cfg [Inhabited Γ] where
  /-- The current machine state. -/
  q : Λ
  /-- The current state of the tape: current symbol, left and right parts. -/
  Tape : Tape Γ


local notation "Cfg₀" => Cfg Γ Λ  -- Porting note (https://github.com/leanprover-community/mathlib4/issues/10750): added this to clean up types.


instance Cfg.inhabited : Inhabited Cfg₀ := ⟨⟨default, default⟩⟩


/-- Execution semantics of the Turing machine. -/
def step (M : Machine₀) : Cfg₀ → Option Cfg₀ :=
  fun ⟨q, T⟩ ↦ (M q T.1).map fun ⟨q', a⟩ ↦ ⟨q', match a with
    | Stmt.move d => T.move d
    | Stmt.write a => T.write a⟩


/-- The statement `Reaches M s₁ s₂` means that `s₂` is obtained
  starting from `s₁` after a finite number of steps from `s₂`. -/
def Reaches (M : Machine₀) : Cfg₀ → Cfg₀ → Prop := ReflTransGen fun a b ↦ b ∈ step M a


/-- The initial configuration. -/
def init (l : List Γ) : Cfg₀ := ⟨default, Tape.mk₁ l⟩


/-- Evaluate a Turing machine on initial input to a final state,
  if it terminates. -/
def eval (M : Machine₀) (l : List Γ) : Part (ListBlank Γ) :=
  (Turing.eval (step M) (init l)).map fun c ↦ c.Tape.right₀


/-- The raw definition of a Turing machine does not require that
  `Γ` and `Λ` are finite, and in practice we will be interested
  in the infinite `Λ` case. We recover instead a notion of
  "effectively finite" Turing machines, which only make use of a
  finite subset of their states. We say that a set `S ⊆ Λ`
  supports a Turing machine `M` if `S` is closed under the
  transition function and contains the initial state. -/
def Supports (M : Machine₀) (S : Set Λ) :=
  default ∈ S ∧ ∀ {q a q' s}, (q', s) ∈ M q a → q ∈ S → q' ∈ S


theorem step_supports (M : Machine₀) {S : Set Λ} (ss : Supports M S) :
    ∀ {c c' : Cfg₀}, c' ∈ step M c → c.q ∈ S → c'.q ∈ S := by
  /-
    Γ : Type u_1
    Λ : Type u_2
    inst✝¹ : Inhabited Λ
    inst✝ : Inhabited Γ
    M : Turing.TM0.Machine Γ Λ
    S : Set Λ
    ss : Turing.TM0.Supports M S
    ⊢ ∀ {c c' : Turing.TM0.Cfg Γ Λ}, Membership.mem (Turing.TM0.step M c) c' → Mem …
  -/
  intro ⟨q, T⟩ c' h₁ h₂
  /-
    Γ : Type u_1
    Λ : Type u_2
    inst✝¹ : Inhabited Λ
    inst✝ : Inhabited Γ
    M : Turing.TM0.Machine Γ Λ
    S : Set Λ
    ss : Turing.TM0.Supports M S
    q : Λ
    T : Turing.Tape Γ
    c' : Turing.TM0.Cfg Γ Λ
    h₁ : Membership.mem (Turing.TM0.step M { q := q, Tape := T }) c'
    h₂ : Membership.mem S { q := q, Tape := T }.q
    ⊢ Membership.mem S c'.q
  -/
  rcases Option.map_eq_some'.1 h₁ with ⟨⟨q', a⟩, h, rfl⟩
  /-
    case intro.mk.intro
    Γ : Type u_1
    Λ : Type u_2
    inst✝¹ : Inhabited Λ
    inst✝ : Inhabited Γ
    M : Turing.TM0.Machine Γ Λ
    S : Set Λ
    ss : Turing.TM0.Supports M S
    q : Λ
    T : Turing.Tape Γ
    h₂ : Membership.mem S { q := q, Tape := T }.q
    q' : Λ
    a : Turing.TM0.Stmt Γ
    h : Eq (M q T.head) (Option.some { fst := q', snd := a })
    h₁ : Membership.mem (Turing.TM0.step M { q := q, Tape := T }) (Turing.TM0.step …
    ⊢ Membership.mem S (Turing.TM0.step.match_2 (fun x => Turing.TM0.Cfg Γ Λ) { fs …
  -/
  exact ss.2 h h₂
  /-
    🎉 no goals
  -/


theorem univ_supports (M : Machine₀) : Supports M Set.univ := by
  /-
    Γ : Type u_1
    Λ : Type u_2
    inst✝ : Inhabited Λ
    M : Turing.TM0.Machine Γ Λ
    ⊢ Turing.TM0.Supports M Set.univ
  -/
                             /-
                               🎉 no goals
                             -/
  constructor <;> intros <;> apply Set.mem_univ
                             /-
                               🎉 no goals
                             -/


/-- Map a TM statement across a function. This does nothing to move statements and maps the write
values. -/
def Stmt.map (f : PointedMap Γ Γ') : Stmt Γ → Stmt Γ'
  | Stmt.move d => Stmt.move d
  | Stmt.write a => Stmt.write (f a)


/-- Map a configuration across a function, given `f : Γ → Γ'` a map of the alphabets and
`g : Λ → Λ'` a map of the machine states. -/
def Cfg.map (f : PointedMap Γ Γ') (g : Λ → Λ') : Cfg Γ Λ → Cfg Γ' Λ'
  | ⟨q, T⟩ => ⟨g q, T.map f⟩


/-- Because the state transition function uses the alphabet and machine states in both the input
and output, to map a machine from one alphabet and machine state space to another we need functions
in both directions, essentially an `Equiv` without the laws. -/
def Machine.map : Machine Γ' Λ'
  | q, l => (M (g₂ q) (f₂ l)).map (Prod.map g₁ (Stmt.map f₁))


theorem Machine.map_step {S : Set Λ} (f₂₁ : Function.RightInverse f₁ f₂)
    (g₂₁ : ∀ q ∈ S, g₂ (g₁ q) = q) :
    ∀ c : Cfg Γ Λ,
      c.q ∈ S → (step M c).map (Cfg.map f₁ g₁) = step (M.map f₁ f₂ g₁ g₂) (Cfg.map f₁ g₁ c)
  | ⟨q, T⟩, h => by
    /-
      Γ : Type u_1
      inst✝³ : Inhabited Γ
      Γ' : Type u_2
      inst✝² : Inhabited Γ'
      Λ : Type u_3
      inst✝¹ : Inhabited Λ
      Λ' : Type u_4
      inst✝ : Inhabited Λ'
      M : Turing.TM0.Machine Γ Λ
      f₁ : Turing.PointedMap Γ Γ'
      f₂ : Turing.PointedMap Γ' Γ
      g₁ : Λ → Λ'
      g₂ : Λ' → Λ
      S : Set Λ
      f₂₁ : Function.RightInverse f₁.f f₂.f
      g₂₁ : ∀ (q : Λ), Membership.mem S q → Eq (g₂ (g₁ q)) q
      q : Λ
      T : Turing.Tape Γ
      h : Membership.mem S { q := q, Tape := T }.q
      ⊢ Eq (Option.map (Turing.TM0.Cfg.map f₁ g₁) (Turing.TM0.step M { q := q, Tape  …
    -/
    unfold step Machine.map Cfg.map
    /-
      Γ : Type u_1
      inst✝³ : Inhabited Γ
      Γ' : Type u_2
      inst✝² : Inhabited Γ'
      Λ : Type u_3
      inst✝¹ : Inhabited Λ
      Λ' : Type u_4
      inst✝ : Inhabited Λ'
      M : Turing.TM0.Machine Γ Λ
      f₁ : Turing.PointedMap Γ Γ'
      f₂ : Turing.PointedMap Γ' Γ
      g₁ : Λ → Λ'
      g₂ : Λ' → Λ
      S : Set Λ
      f₂₁ : Function.RightInverse f₁.f f₂.f
      g₂₁ : ∀ (q : Λ), Membership.mem S q → Eq (g₂ (g₁ q)) q
      q : Λ
      T : Turing.Tape Γ
      h : Membership.mem S { q := q, Tape := T }.q
      ⊢ Eq (Option.map (fun x => Turing.TM0.Cfg.map.match_1 (fun x => Turing.TM0.Cfg …
    -/
    simp only [Turing.Tape.map_fst, g₂₁ q h, f₂₁ _]
    /-
      Γ : Type u_1
      inst✝³ : Inhabited Γ
      Γ' : Type u_2
      inst✝² : Inhabited Γ'
      Λ : Type u_3
      inst✝¹ : Inhabited Λ
      Λ' : Type u_4
      inst✝ : Inhabited Λ'
      M : Turing.TM0.Machine Γ Λ
      f₁ : Turing.PointedMap Γ Γ'
      f₂ : Turing.PointedMap Γ' Γ
      g₁ : Λ → Λ'
      g₂ : Λ' → Λ
      S : Set Λ
      f₂₁ : Function.RightInverse f₁.f f₂.f
      g₂₁ : ∀ (q : Λ), Membership.mem S q → Eq (g₂ (g₁ q)) q
      q : Λ
      T : Turing.Tape Γ
      h : Membership.mem S { q := q, Tape := T }.q
      ⊢ Eq (Option.map (fun x => { q := g₁ x.q, Tape := Turing.Tape.map f₁ x.Tape }) …
    -/
    rcases M q T.1 with (_ | ⟨q', d | a⟩); · rfl
                                             /-
                                               🎉 no goals
                                             -/
      /-
        case some.mk.move
        Γ : Type u_1
        inst✝³ : Inhabited Γ
        Γ' : Type u_2
        inst✝² : Inhabited Γ'
        Λ : Type u_3
        inst✝¹ : Inhabited Λ
        Λ' : Type u_4
        inst✝ : Inhabited Λ'
        M : Turing.TM0.Machine Γ Λ
        f₁ : Turing.PointedMap Γ Γ'
        f₂ : Turing.PointedMap Γ' Γ
        g₁ : Λ → Λ'
        g₂ : Λ' → Λ
        S : Set Λ
        f₂₁ : Function.RightInverse f₁.f f₂.f
        g₂₁ : ∀ (q : Λ), Membership.mem S q → Eq (g₂ (g₁ q)) q
        q : Λ
        T : Turing.Tape Γ
        h : Membership.mem S { q := q, Tape := T }.q
        q' : Λ
        d : Turing.Dir
        ⊢ Eq (Option.map (fun x => { q := g₁ x.q, Tape := Turing.Tape.map f₁ x.Tape }) …
      -/
    · simp only [step, Cfg.map, Option.map_some', Tape.map_move f₁]
      /-
        case some.mk.move
        Γ : Type u_1
        inst✝³ : Inhabited Γ
        Γ' : Type u_2
        inst✝² : Inhabited Γ'
        Λ : Type u_3
        inst✝¹ : Inhabited Λ
        Λ' : Type u_4
        inst✝ : Inhabited Λ'
        M : Turing.TM0.Machine Γ Λ
        f₁ : Turing.PointedMap Γ Γ'
        f₂ : Turing.PointedMap Γ' Γ
        g₁ : Λ → Λ'
        g₂ : Λ' → Λ
        S : Set Λ
        f₂₁ : Function.RightInverse f₁.f f₂.f
        g₂₁ : ∀ (q : Λ), Membership.mem S q → Eq (g₂ (g₁ q)) q
        q : Λ
        T : Turing.Tape Γ
        h : Membership.mem S { q := q, Tape := T }.q
        q' : Λ
        d : Turing.Dir
        ⊢ Eq (Option.some { q := g₁ q', Tape := Turing.Tape.move d (Turing.Tape.map f₁ …
      -/
      rfl
      /-
        🎉 no goals
      -/
      /-
        case some.mk.write
        Γ : Type u_1
        inst✝³ : Inhabited Γ
        Γ' : Type u_2
        inst✝² : Inhabited Γ'
        Λ : Type u_3
        inst✝¹ : Inhabited Λ
        Λ' : Type u_4
        inst✝ : Inhabited Λ'
        M : Turing.TM0.Machine Γ Λ
        f₁ : Turing.PointedMap Γ Γ'
        f₂ : Turing.PointedMap Γ' Γ
        g₁ : Λ → Λ'
        g₂ : Λ' → Λ
        S : Set Λ
        f₂₁ : Function.RightInverse f₁.f f₂.f
        g₂₁ : ∀ (q : Λ), Membership.mem S q → Eq (g₂ (g₁ q)) q
        q : Λ
        T : Turing.Tape Γ
        h : Membership.mem S { q := q, Tape := T }.q
        q' : Λ
        a : Γ
        ⊢ Eq (Option.map (fun x => { q := g₁ x.q, Tape := Turing.Tape.map f₁ x.Tape }) …
      -/
    · simp only [step, Cfg.map, Option.map_some', Tape.map_write]
      /-
        case some.mk.write
        Γ : Type u_1
        inst✝³ : Inhabited Γ
        Γ' : Type u_2
        inst✝² : Inhabited Γ'
        Λ : Type u_3
        inst✝¹ : Inhabited Λ
        Λ' : Type u_4
        inst✝ : Inhabited Λ'
        M : Turing.TM0.Machine Γ Λ
        f₁ : Turing.PointedMap Γ Γ'
        f₂ : Turing.PointedMap Γ' Γ
        g₁ : Λ → Λ'
        g₂ : Λ' → Λ
        S : Set Λ
        f₂₁ : Function.RightInverse f₁.f f₂.f
        g₂₁ : ∀ (q : Λ), Membership.mem S q → Eq (g₂ (g₁ q)) q
        q : Λ
        T : Turing.Tape Γ
        h : Membership.mem S { q := q, Tape := T }.q
        q' : Λ
        a : Γ
        ⊢ Eq (Option.some { q := g₁ q', Tape := Turing.Tape.write (f₁.f a) (Turing.Tap …
      -/
      rfl
      /-
        🎉 no goals
      -/


theorem map_init (g₁ : PointedMap Λ Λ') (l : List Γ) : (init l).map f₁ g₁ = init (l.map f₁) :=
  congr (congr_arg Cfg.mk g₁.map_pt) (Tape.map_mk₁ _ _)


theorem Machine.map_respects (g₁ : PointedMap Λ Λ') (g₂ : Λ' → Λ) {S} (ss : Supports M S)
    (f₂₁ : Function.RightInverse f₁ f₂) (g₂₁ : ∀ q ∈ S, g₂ (g₁ q) = q) :
    Respects (step M) (step (M.map f₁ f₂ g₁ g₂)) fun a b ↦ a.q ∈ S ∧ Cfg.map f₁ g₁ a = b := by
  /-
    Γ : Type u_1
    inst✝³ : Inhabited Γ
    Γ' : Type u_2
    inst✝² : Inhabited Γ'
    Λ : Type u_3
    inst✝¹ : Inhabited Λ
    Λ' : Type u_4
    inst✝ : Inhabited Λ'
    M : Turing.TM0.Machine Γ Λ
    f₁ : Turing.PointedMap Γ Γ'
    f₂ : Turing.PointedMap Γ' Γ
    g₁ : Turing.PointedMap Λ Λ'
    g₂ : Λ' → Λ
    S : Set Λ
    ss : Turing.TM0.Supports M S
    f₂₁ : Function.RightInverse f₁.f f₂.f
    g₂₁ : ∀ (q : Λ), Membership.mem S q → Eq (g₂ (g₁.f q)) q
    ⊢ Turing.Respects (Turing.TM0.step M) (Turing.TM0.step (M.map f₁ f₂ g₁.f g₂))  …
  -/
  intro c _ ⟨cs, rfl⟩
  /-
    Γ : Type u_1
    inst✝³ : Inhabited Γ
    Γ' : Type u_2
    inst✝² : Inhabited Γ'
    Λ : Type u_3
    inst✝¹ : Inhabited Λ
    Λ' : Type u_4
    inst✝ : Inhabited Λ'
    M : Turing.TM0.Machine Γ Λ
    f₁ : Turing.PointedMap Γ Γ'
    f₂ : Turing.PointedMap Γ' Γ
    g₁ : Turing.PointedMap Λ Λ'
    g₂ : Λ' → Λ
    S : Set Λ
    ss : Turing.TM0.Supports M S
    f₂₁ : Function.RightInverse f₁.f f₂.f
    g₂₁ : ∀ (q : Λ), Membership.mem S q → Eq (g₂ (g₁.f q)) q
    c : Turing.TM0.Cfg Γ Λ
    a₂✝ : Turing.TM0.Cfg Γ' Λ'
    cs : Membership.mem S c.q
    ⊢ Turing.Respects.match_1 (fun x => Prop) (Turing.TM0.step M c) (fun b₁ => Exi …
  -/
  cases e : step M c
    /-
      case none
      Γ : Type u_1
      inst✝³ : Inhabited Γ
      Γ' : Type u_2
      inst✝² : Inhabited Γ'
      Λ : Type u_3
      inst✝¹ : Inhabited Λ
      Λ' : Type u_4
      inst✝ : Inhabited Λ'
      M : Turing.TM0.Machine Γ Λ
      f₁ : Turing.PointedMap Γ Γ'
      f₂ : Turing.PointedMap Γ' Γ
      g₁ : Turing.PointedMap Λ Λ'
      g₂ : Λ' → Λ
      S : Set Λ
      ss : Turing.TM0.Supports M S
      f₂₁ : Function.RightInverse f₁.f f₂.f
      g₂₁ : ∀ (q : Λ), Membership.mem S q → Eq (g₂ (g₁.f q)) q
      c : Turing.TM0.Cfg Γ Λ
      a₂✝ : Turing.TM0.Cfg Γ' Λ'
      cs : Membership.mem S c.q
      e : Eq (Turing.TM0.step M c) Option.none
      ⊢ Turing.Respects.match_1 (fun x => Prop) Option.none (fun b₁ => Exists fun b₂ …
    -/
  · rw [← M.map_step f₁ f₂ g₁ g₂ f₂₁ g₂₁ _ cs, e]
    /-
      case none
      Γ : Type u_1
      inst✝³ : Inhabited Γ
      Γ' : Type u_2
      inst✝² : Inhabited Γ'
      Λ : Type u_3
      inst✝¹ : Inhabited Λ
      Λ' : Type u_4
      inst✝ : Inhabited Λ'
      M : Turing.TM0.Machine Γ Λ
      f₁ : Turing.PointedMap Γ Γ'
      f₂ : Turing.PointedMap Γ' Γ
      g₁ : Turing.PointedMap Λ Λ'
      g₂ : Λ' → Λ
      S : Set Λ
      ss : Turing.TM0.Supports M S
      f₂₁ : Function.RightInverse f₁.f f₂.f
      g₂₁ : ∀ (q : Λ), Membership.mem S q → Eq (g₂ (g₁.f q)) q
      c : Turing.TM0.Cfg Γ Λ
      a₂✝ : Turing.TM0.Cfg Γ' Λ'
      cs : Membership.mem S c.q
      e : Eq (Turing.TM0.step M c) Option.none
      ⊢ Turing.Respects.match_1 (fun x => Prop) Option.none (fun b₁ => Exists fun b₂ …
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case some
      Γ : Type u_1
      inst✝³ : Inhabited Γ
      Γ' : Type u_2
      inst✝² : Inhabited Γ'
      Λ : Type u_3
      inst✝¹ : Inhabited Λ
      Λ' : Type u_4
      inst✝ : Inhabited Λ'
      M : Turing.TM0.Machine Γ Λ
      f₁ : Turing.PointedMap Γ Γ'
      f₂ : Turing.PointedMap Γ' Γ
      g₁ : Turing.PointedMap Λ Λ'
      g₂ : Λ' → Λ
      S : Set Λ
      ss : Turing.TM0.Supports M S
      f₂₁ : Function.RightInverse f₁.f f₂.f
      g₂₁ : ∀ (q : Λ), Membership.mem S q → Eq (g₂ (g₁.f q)) q
      c : Turing.TM0.Cfg Γ Λ
      a₂✝ : Turing.TM0.Cfg Γ' Λ'
      cs : Membership.mem S c.q
      val✝ : Turing.TM0.Cfg Γ Λ
      e : Eq (Turing.TM0.step M c) (Option.some val✝)
      ⊢ Turing.Respects.match_1 (fun x => Prop) (Option.some val✝) (fun b₁ => Exists …
    -/
  · refine ⟨_, ⟨step_supports M ss e cs, rfl⟩, TransGen.single ?_⟩
    /-
      case some
      Γ : Type u_1
      inst✝³ : Inhabited Γ
      Γ' : Type u_2
      inst✝² : Inhabited Γ'
      Λ : Type u_3
      inst✝¹ : Inhabited Λ
      Λ' : Type u_4
      inst✝ : Inhabited Λ'
      M : Turing.TM0.Machine Γ Λ
      f₁ : Turing.PointedMap Γ Γ'
      f₂ : Turing.PointedMap Γ' Γ
      g₁ : Turing.PointedMap Λ Λ'
      g₂ : Λ' → Λ
      S : Set Λ
      ss : Turing.TM0.Supports M S
      f₂₁ : Function.RightInverse f₁.f f₂.f
      g₂₁ : ∀ (q : Λ), Membership.mem S q → Eq (g₂ (g₁.f q)) q
      c : Turing.TM0.Cfg Γ Λ
      a₂✝ : Turing.TM0.Cfg Γ' Λ'
      cs : Membership.mem S c.q
      val✝ : Turing.TM0.Cfg Γ Λ
      e : Eq (Turing.TM0.step M c) (Option.some val✝)
      ⊢ Membership.mem (Turing.TM0.step (M.map f₁ f₂ g₁.f g₂) (Turing.TM0.Cfg.map f₁ …
    -/
    rw [← M.map_step f₁ f₂ g₁ g₂ f₂₁ g₂₁ _ cs, e]
    /-
      case some
      Γ : Type u_1
      inst✝³ : Inhabited Γ
      Γ' : Type u_2
      inst✝² : Inhabited Γ'
      Λ : Type u_3
      inst✝¹ : Inhabited Λ
      Λ' : Type u_4
      inst✝ : Inhabited Λ'
      M : Turing.TM0.Machine Γ Λ
      f₁ : Turing.PointedMap Γ Γ'
      f₂ : Turing.PointedMap Γ' Γ
      g₁ : Turing.PointedMap Λ Λ'
      g₂ : Λ' → Λ
      S : Set Λ
      ss : Turing.TM0.Supports M S
      f₂₁ : Function.RightInverse f₁.f f₂.f
      g₂₁ : ∀ (q : Λ), Membership.mem S q → Eq (g₂ (g₁.f q)) q
      c : Turing.TM0.Cfg Γ Λ
      a₂✝ : Turing.TM0.Cfg Γ' Λ'
      cs : Membership.mem S c.q
      val✝ : Turing.TM0.Cfg Γ Λ
      e : Eq (Turing.TM0.step M c) (Option.some val✝)
      ⊢ Membership.mem (Option.map (Turing.TM0.Cfg.map f₁ g₁.f) (Option.some val✝))  …
    -/
    rfl
    /-
      🎉 no goals
    -/


/-- The TM1 model is a simplification and extension of TM0
  (Post-Turing model) in the direction of Wang B-machines. The machine's
  internal state is extended with a (finite) store `σ` of variables
  that may be accessed and updated at any time.
  A machine is given by a `Λ` indexed set of procedures or functions.
  Each function has a body which is a `Stmt`, which can either be a
  `move` or `write` command, a `branch` (if statement based on the
  current tape value), a `load` (set the variable value),
  a `goto` (call another function), or `halt`. Note that here
  most statements do not have labels; `goto` commands can only
  go to a new function. All commands have access to the variable value
  and current tape value. -/
inductive Stmt
  | move : Dir → Stmt → Stmt
  | write : (Γ → σ → Γ) → Stmt → Stmt
  | load : (Γ → σ → σ) → Stmt → Stmt
  | branch : (Γ → σ → Bool) → Stmt → Stmt → Stmt
  | goto : (Γ → σ → Λ) → Stmt
  | halt : Stmt


local notation "Stmt₁" => Stmt Γ Λ σ  -- Porting note (https://github.com/leanprover-community/mathlib4/issues/10750): added this to clean up types.


instance Stmt.inhabited : Inhabited Stmt₁ := ⟨halt⟩


/-- The configuration of a TM1 machine is given by the currently
  evaluating statement, the variable store value, and the tape. -/
structure Cfg [Inhabited Γ] where
  /-- The statement (if any) which is currently evaluated -/
  l : Option Λ
  /-- The current value of the variable store -/
  var : σ
  /-- The current state of the tape -/
  Tape : Tape Γ


local notation "Cfg₁" => Cfg Γ Λ σ  -- Porting note (https://github.com/leanprover-community/mathlib4/issues/10750): added this to clean up types.


instance Cfg.inhabited [Inhabited Γ] [Inhabited σ] : Inhabited Cfg₁ :=
  ⟨⟨default, default, default⟩⟩


/-- The semantics of TM1 evaluation. -/
def stepAux [Inhabited Γ] : Stmt₁ → σ → Tape Γ → Cfg₁
  | move d q, v, T => stepAux q v (T.move d)
  | write a q, v, T => stepAux q v (T.write (a T.1 v))
  | load s q, v, T => stepAux q (s T.1 v) T
  | branch p q₁ q₂, v, T => cond (p T.1 v) (stepAux q₁ v T) (stepAux q₂ v T)
  | goto l, v, T => ⟨some (l T.1 v), v, T⟩
  | halt, v, T => ⟨none, v, T⟩


/-- The state transition function. -/
def step [Inhabited Γ] (M : Λ → Stmt₁) : Cfg₁ → Option Cfg₁
  | ⟨none, _, _⟩ => none
  | ⟨some l, v, T⟩ => some (stepAux (M l) v T)


/-- A set `S` of labels supports the statement `q` if all the `goto`
  statements in `q` refer only to other functions in `S`. -/
def SupportsStmt (S : Finset Λ) : Stmt₁ → Prop
  | move _ q => SupportsStmt S q
  | write _ q => SupportsStmt S q
  | load _ q => SupportsStmt S q
  | branch _ q₁ q₂ => SupportsStmt S q₁ ∧ SupportsStmt S q₂
  | goto l => ∀ a v, l a v ∈ S
  | halt => True


/-- The subterm closure of a statement. -/
noncomputable def stmts₁ : Stmt₁ → Finset Stmt₁
  | Q@(move _ q) => insert Q (stmts₁ q)
  | Q@(write _ q) => insert Q (stmts₁ q)
  | Q@(load _ q) => insert Q (stmts₁ q)
  | Q@(branch _ q₁ q₂) => insert Q (stmts₁ q₁ ∪ stmts₁ q₂)
  | Q => {Q}


theorem stmts₁_self {q : Stmt₁} : q ∈ stmts₁ q := by
  /-
    Γ : Type u_1
    Λ : Type u_2
    σ : Type u_3
    q : Turing.TM1.Stmt Γ Λ σ
    ⊢ Membership.mem (Turing.TM1.stmts₁ q) q
  -/
              /-
                🎉 no goals
              -/
              /-
                🎉 no goals
              -/
              /-
                🎉 no goals
              -/
              /-
                🎉 no goals
              -/
              /-
                🎉 no goals
              -/
  cases q <;> simp only [stmts₁, Finset.mem_insert_self, Finset.mem_singleton_self]
              /-
                🎉 no goals
              -/


theorem stmts₁_trans {q₁ q₂ : Stmt₁} : q₁ ∈ stmts₁ q₂ → stmts₁ q₁ ⊆ stmts₁ q₂ := by
  /-
    Γ : Type u_1
    Λ : Type u_2
    σ : Type u_3
    q₁ q₂ : Turing.TM1.Stmt Γ Λ σ
    ⊢ Membership.mem (Turing.TM1.stmts₁ q₂) q₁ → HasSubset.Subset (Turing.TM1.stmt …
  -/
  intro h₁₂ q₀ h₀₁
  induction q₂ with (
    simp only [stmts₁] at h₁₂ ⊢
    simp only [Finset.mem_insert, Finset.mem_union, Finset.mem_singleton] at h₁₂)
  | branch p q₁ q₂ IH₁ IH₂ =>
    rcases h₁₂ with (rfl | h₁₂ | h₁₂)
    · unfold stmts₁ at h₀₁
      exact h₀₁
    · exact Finset.mem_insert_of_mem (Finset.mem_union_left _ <| IH₁ h₁₂)
    · exact Finset.mem_insert_of_mem (Finset.mem_union_right _ <| IH₂ h₁₂)
  | goto l => subst h₁₂; exact h₀₁
  | halt => subst h₁₂; exact h₀₁
  | _ _ q IH =>
    rcases h₁₂ with rfl | h₁₂
    · exact h₀₁
    · exact Finset.mem_insert_of_mem (IH h₁₂)


theorem stmts₁_supportsStmt_mono {S : Finset Λ} {q₁ q₂ : Stmt₁} (h : q₁ ∈ stmts₁ q₂)
    (hs : SupportsStmt S q₂) : SupportsStmt S q₁ := by
  induction q₂ with
    simp only [stmts₁, SupportsStmt, Finset.mem_insert, Finset.mem_union, Finset.mem_singleton]
      at h hs
  | branch p q₁ q₂ IH₁ IH₂ => rcases h with (rfl | h | h); exacts [hs, IH₁ h hs.1, IH₂ h hs.2]
  | goto l => subst h; exact hs
  | halt => subst h; trivial
  | _ _ q IH => rcases h with (rfl | h) <;> [exact hs; exact IH h hs]


/-- The set of all statements in a Turing machine, plus one extra value `none` representing the
halt state. This is used in the TM1 to TM0 reduction. -/
noncomputable def stmts (M : Λ → Stmt₁) (S : Finset Λ) : Finset (Option Stmt₁) :=
  Finset.insertNone (S.biUnion fun q ↦ stmts₁ (M q))


theorem stmts_trans {M : Λ → Stmt₁} {S : Finset Λ} {q₁ q₂ : Stmt₁} (h₁ : q₁ ∈ stmts₁ q₂) :
    some q₂ ∈ stmts M S → some q₁ ∈ stmts M S := by
  simp only [stmts, Finset.mem_insertNone, Finset.mem_biUnion, Option.mem_def, Option.some.injEq,
    forall_eq', exists_imp, and_imp]
  /-
    Γ : Type u_1
    Λ : Type u_2
    σ : Type u_3
    M : Λ → Turing.TM1.Stmt Γ Λ σ
    S : Finset Λ
    q₁ q₂ : Turing.TM1.Stmt Γ Λ σ
    h₁ : Membership.mem (Turing.TM1.stmts₁ q₂) q₁
    ⊢ ∀ (x : Λ), Membership.mem S x → Membership.mem (Turing.TM1.stmts₁ (M x)) q₂  …
  -/
  exact fun l ls h₂ ↦ ⟨_, ls, stmts₁_trans h₂ h₁⟩
  /-
    🎉 no goals
  -/


/-- A set `S` of labels supports machine `M` if all the `goto`
  statements in the functions in `S` refer only to other functions
  in `S`. -/
def Supports (M : Λ → Stmt₁) (S : Finset Λ) :=
  default ∈ S ∧ ∀ q ∈ S, SupportsStmt S (M q)


theorem stmts_supportsStmt {M : Λ → Stmt₁} {S : Finset Λ} {q : Stmt₁} (ss : Supports M S) :
    some q ∈ stmts M S → SupportsStmt S q := by
  simp only [stmts, Finset.mem_insertNone, Finset.mem_biUnion, Option.mem_def, Option.some.injEq,
    forall_eq', exists_imp, and_imp]
  /-
    Γ : Type u_1
    Λ : Type u_2
    σ : Type u_3
    inst✝ : Inhabited Λ
    M : Λ → Turing.TM1.Stmt Γ Λ σ
    S : Finset Λ
    q : Turing.TM1.Stmt Γ Λ σ
    ss : Turing.TM1.Supports M S
    ⊢ ∀ (x : Λ), Membership.mem S x → Membership.mem (Turing.TM1.stmts₁ (M x)) q → …
  -/
  exact fun l ls h ↦ stmts₁_supportsStmt_mono h (ss.2 _ ls)
  /-
    🎉 no goals
  -/


theorem step_supports (M : Λ → Stmt₁) {S : Finset Λ} (ss : Supports M S) :
    ∀ {c c' : Cfg₁}, c' ∈ step M c → c.l ∈ Finset.insertNone S → c'.l ∈ Finset.insertNone S
  | ⟨some l₁, v, T⟩, c', h₁, h₂ => by
    /-
      Γ : Type u_1
      Λ : Type u_2
      σ : Type u_3
      inst✝¹ : Inhabited Λ
      inst✝ : Inhabited Γ
      M : Λ → Turing.TM1.Stmt Γ Λ σ
      S : Finset Λ
      ss : Turing.TM1.Supports M S
      l₁ : Λ
      v : σ
      T : Turing.Tape Γ
      c' : Turing.TM1.Cfg Γ Λ σ
      h₁ : Membership.mem (Turing.TM1.step M { l := Option.some l₁, var := v, Tape : …
      h₂ : Membership.mem (Finset.insertNone S) { l := Option.some l₁, var := v, Tap …
      ⊢ Membership.mem (Finset.insertNone S) c'.l
    -/
    replace h₂ := ss.2 _ (Finset.some_mem_insertNone.1 h₂)
    /-
      Γ : Type u_1
      Λ : Type u_2
      σ : Type u_3
      inst✝¹ : Inhabited Λ
      inst✝ : Inhabited Γ
      M : Λ → Turing.TM1.Stmt Γ Λ σ
      S : Finset Λ
      ss : Turing.TM1.Supports M S
      l₁ : Λ
      v : σ
      T : Turing.Tape Γ
      c' : Turing.TM1.Cfg Γ Λ σ
      h₁ : Membership.mem (Turing.TM1.step M { l := Option.some l₁, var := v, Tape : …
      h₂ : Turing.TM1.SupportsStmt S (M l₁)
      ⊢ Membership.mem (Finset.insertNone S) c'.l
    -/
    simp only [step, Option.mem_def, Option.some.injEq] at h₁; subst c'
    /-
      Γ : Type u_1
      Λ : Type u_2
      σ : Type u_3
      inst✝¹ : Inhabited Λ
      inst✝ : Inhabited Γ
      M : Λ → Turing.TM1.Stmt Γ Λ σ
      S : Finset Λ
      ss : Turing.TM1.Supports M S
      l₁ : Λ
      v : σ
      T : Turing.Tape Γ
      h₂ : Turing.TM1.SupportsStmt S (M l₁)
      ⊢ Membership.mem (Finset.insertNone S) (Turing.TM1.stepAux (M l₁) v T).l
    -/
    revert h₂; induction M l₁ generalizing v T with intro hs
    | branch p q₁' q₂' IH₁ IH₂ =>
      unfold stepAux; cases p T.1 v
      · exact IH₂ _ _ hs.2
      · exact IH₁ _ _ hs.1
    | goto => exact Finset.some_mem_insertNone.2 (hs _ _)
    | halt => apply Multiset.mem_cons_self
    | _ _ q IH => exact IH _ _ hs


/-- The initial state, given a finite input that is placed on the tape starting at the TM head and
going to the right. -/
def init (l : List Γ) : Cfg₁ :=
  ⟨some default, default, Tape.mk₁ l⟩


/-- Evaluate a TM to completion, resulting in an output list on the tape (with an indeterminate
number of blanks on the end). -/
def eval (M : Λ → Stmt₁) (l : List Γ) : Part (ListBlank Γ) :=
  (Turing.eval (step M) (init l)).map fun c ↦ c.Tape.right₀


local notation "Stmt₁" => TM1.Stmt Γ Λ σ


local notation "Cfg₁" => TM1.Cfg Γ Λ σ


local notation "Stmt₀" => TM0.Stmt Γ


set_option linter.unusedVariables false in
/-- The base machine state space is a pair of an `Option Stmt₁` representing the current program
to be executed, or `none` for the halt state, and a `σ` which is the local state (stored in the TM,
not the tape). Because there are an infinite number of programs, this state space is infinite, but
for a finitely supported TM1 machine and a finite type `σ`, only finitely many of these states are
reachable. -/
@[nolint unusedArguments] -- We need the M assumption
def Λ' (M : Λ → TM1.Stmt Γ Λ σ) :=
  Option Stmt₁ × σ


local notation "Λ'₁₀" => Λ' M -- Porting note (https://github.com/leanprover-community/mathlib4/issues/10750): added this to clean up types.


instance : Inhabited Λ'₁₀ :=
  ⟨(some (M default), default)⟩


/-- The core TM1 → TM0 translation function. Here `s` is the current value on the tape, and the
`Stmt₁` is the TM1 statement to translate, with local state `v : σ`. We evaluate all regular
instructions recursively until we reach either a `move` or `write` command, or a `goto`; in the
latter case we emit a dummy `write s` step and transition to the new target location. -/
def trAux (s : Γ) : Stmt₁ → σ → Λ'₁₀ × Stmt₀
  | TM1.Stmt.move d q, v => ((some q, v), move d)
  | TM1.Stmt.write a q, v => ((some q, v), write (a s v))
  | TM1.Stmt.load a q, v => trAux s q (a s v)
  | TM1.Stmt.branch p q₁ q₂, v => cond (p s v) (trAux s q₁ v) (trAux s q₂ v)
  | TM1.Stmt.goto l, v => ((some (M (l s v)), v), write s)
  | TM1.Stmt.halt, v => ((none, v), write s)


local notation "Cfg₁₀" => TM0.Cfg Γ Λ'₁₀


/-- The translated TM0 machine (given the TM1 machine input). -/
def tr : TM0.Machine Γ Λ'₁₀
  | (none, _), _ => none
  | (some q, v), s => some (trAux M s q v)


/-- Translate configurations from TM1 to TM0. -/
def trCfg [Inhabited Γ] : Cfg₁ → Cfg₁₀
  | ⟨l, v, T⟩ => ⟨(l.map M, v), T⟩


theorem tr_respects [Inhabited Γ] :
    Respects (TM1.step M) (TM0.step (tr M)) fun (c₁ : Cfg₁) (c₂ : Cfg₁₀) ↦ trCfg M c₁ = c₂ :=
  fun_respects.2 fun ⟨l₁, v, T⟩ ↦ by
    /-
      Γ : Type u_1
      Λ : Type u_2
      inst✝² : Inhabited Λ
      σ : Type u_3
      inst✝¹ : Inhabited σ
      M : Λ → Turing.TM1.Stmt Γ Λ σ
      inst✝ : Inhabited Γ
      x✝ : Turing.TM1.Cfg Γ Λ σ
      l₁ : Option Λ
      v : σ
      T : Turing.Tape Γ
      ⊢ Turing.FRespects (Turing.TM0.step (Turing.TM1to0.tr M)) (Turing.TM1to0.trCfg …
    -/
    cases' l₁ with l₁; · exact rfl
                         /-
                           🎉 no goals
                         -/
    /-
      case some
      Γ : Type u_1
      Λ : Type u_2
      inst✝² : Inhabited Λ
      σ : Type u_3
      inst✝¹ : Inhabited σ
      M : Λ → Turing.TM1.Stmt Γ Λ σ
      inst✝ : Inhabited Γ
      x✝ : Turing.TM1.Cfg Γ Λ σ
      v : σ
      T : Turing.Tape Γ
      l₁ : Λ
      ⊢ Turing.FRespects (Turing.TM0.step (Turing.TM1to0.tr M)) (Turing.TM1to0.trCfg …
    -/
    simp only [trCfg, TM1.step, FRespects, Option.map]
    induction M l₁ generalizing v T with
    | move _ _ IH => exact TransGen.head rfl (IH _ _)
    | write _ _ IH => exact TransGen.head rfl (IH _ _)
    | load _ _ IH => exact (reaches₁_eq (by rfl)).2 (IH _ _)
    | branch p _ _ IH₁ IH₂ =>
      unfold TM1.stepAux; cases e : p T.1 v
      · exact (reaches₁_eq (by simp only [TM0.step, tr, trAux, e]; rfl)).2 (IH₂ _ _)
      · exact (reaches₁_eq (by simp only [TM0.step, tr, trAux, e]; rfl)).2 (IH₁ _ _)
    | _ =>
      exact TransGen.single (congr_arg some (congr (congr_arg TM0.Cfg.mk rfl) (Tape.write_self T)))


theorem tr_eval [Inhabited Γ] (l : List Γ) : TM0.eval (tr M) l = TM1.eval M l :=
  (congr_arg _ (tr_eval' _ _ _ (tr_respects M) ⟨some _, _, _⟩)).trans
    (by
      /-
        Γ : Type u_1
        Λ : Type u_2
        inst✝² : Inhabited Λ
        σ : Type u_3
        inst✝¹ : Inhabited σ
        M : Λ → Turing.TM1.Stmt Γ Λ σ
        inst✝ : Inhabited Γ
        l : List Γ
        ⊢ Eq (Part.map (fun c => c.Tape.right₀) (Functor.map (Turing.TM1to0.trCfg M) ( …
      -/
      rw [Part.map_eq_map, Part.map_map, TM1.eval]
      /-
        Γ : Type u_1
        Λ : Type u_2
        inst✝² : Inhabited Λ
        σ : Type u_3
        inst✝¹ : Inhabited σ
        M : Λ → Turing.TM1.Stmt Γ Λ σ
        inst✝ : Inhabited Γ
        l : List Γ
        ⊢ Eq (Part.map (Function.comp (fun c => c.Tape.right₀) (Turing.TM1to0.trCfg M) …
      -/
      congr with ⟨⟩)
      /-
        🎉 no goals
      -/


/-- Given a finite set of accessible `Λ` machine states, there is a finite set of accessible
machine states in the target (even though the type `Λ'` is infinite). -/
noncomputable def trStmts (S : Finset Λ) : Finset Λ'₁₀ :=
  (TM1.stmts M S) ×ˢ Finset.univ


attribute [local simp] TM1.stmts₁_self


theorem tr_supports {S : Finset Λ} (ss : TM1.Supports M S) :
    TM0.Supports (tr M) ↑(trStmts M S) := by
  /-
    Γ : Type u_1
    Λ : Type u_2
    inst✝² : Inhabited Λ
    σ : Type u_3
    inst✝¹ : Inhabited σ
    M : Λ → Turing.TM1.Stmt Γ Λ σ
    inst✝ : Fintype σ
    S : Finset Λ
    ss : Turing.TM1.Supports M S
    ⊢ Turing.TM0.Supports (Turing.TM1to0.tr M) ↑(Turing.TM1to0.trStmts M S)
  -/
  constructor
    /-
      case left
      Γ : Type u_1
      Λ : Type u_2
      inst✝² : Inhabited Λ
      σ : Type u_3
      inst✝¹ : Inhabited σ
      M : Λ → Turing.TM1.Stmt Γ Λ σ
      inst✝ : Fintype σ
      S : Finset Λ
      ss : Turing.TM1.Supports M S
      ⊢ Membership.mem (↑(Turing.TM1to0.trStmts M S)) Inhabited.default
    -/
  · apply Finset.mem_product.2
    /-
      case left
      Γ : Type u_1
      Λ : Type u_2
      inst✝² : Inhabited Λ
      σ : Type u_3
      inst✝¹ : Inhabited σ
      M : Λ → Turing.TM1.Stmt Γ Λ σ
      inst✝ : Fintype σ
      S : Finset Λ
      ss : Turing.TM1.Supports M S
      ⊢ And (Membership.mem (Turing.TM1.stmts M S) Inhabited.default.1) (Membership. …
    -/
    constructor
    · simp only [default, TM1.stmts, Finset.mem_insertNone, Option.mem_def, Option.some_inj,
        forall_eq', Finset.mem_biUnion]
      /-
        case left.left
        Γ : Type u_1
        Λ : Type u_2
        inst✝² : Inhabited Λ
        σ : Type u_3
        inst✝¹ : Inhabited σ
        M : Λ → Turing.TM1.Stmt Γ Λ σ
        inst✝ : Fintype σ
        S : Finset Λ
        ss : Turing.TM1.Supports M S
        ⊢ Exists fun a => And (Membership.mem S a) (Membership.mem (Turing.TM1.stmts₁  …
      -/
      exact ⟨_, ss.1, TM1.stmts₁_self⟩
      /-
        🎉 no goals
      -/
      /-
        case left.right
        Γ : Type u_1
        Λ : Type u_2
        inst✝² : Inhabited Λ
        σ : Type u_3
        inst✝¹ : Inhabited σ
        M : Λ → Turing.TM1.Stmt Γ Λ σ
        inst✝ : Fintype σ
        S : Finset Λ
        ss : Turing.TM1.Supports M S
        ⊢ Membership.mem Finset.univ Inhabited.default.2
      -/
    · apply Finset.mem_univ
      /-
        🎉 no goals
      -/
    /-
      case right
      Γ : Type u_1
      Λ : Type u_2
      inst✝² : Inhabited Λ
      σ : Type u_3
      inst✝¹ : Inhabited σ
      M : Λ → Turing.TM1.Stmt Γ Λ σ
      inst✝ : Fintype σ
      S : Finset Λ
      ss : Turing.TM1.Supports M S
      ⊢ ∀ {q : Turing.TM1to0.Λ' M} {a : Γ} {q' : Turing.TM1to0.Λ' M} {s : Turing.TM0 …
    -/
  · intro q a q' s h₁ h₂
    /-
      case right
      Γ : Type u_1
      Λ : Type u_2
      inst✝² : Inhabited Λ
      σ : Type u_3
      inst✝¹ : Inhabited σ
      M : Λ → Turing.TM1.Stmt Γ Λ σ
      inst✝ : Fintype σ
      S : Finset Λ
      ss : Turing.TM1.Supports M S
      q : Turing.TM1to0.Λ' M
      a : Γ
      q' : Turing.TM1to0.Λ' M
      s : Turing.TM0.Stmt Γ
      h₁ : Membership.mem (Turing.TM1to0.tr M q a) { fst := q', snd := s }
      h₂ : Membership.mem (↑(Turing.TM1to0.trStmts M S)) q
      ⊢ Membership.mem (↑(Turing.TM1to0.trStmts M S)) q'
    -/
    rcases q with ⟨_ | q, v⟩; · cases h₁
                                /-
                                  🎉 no goals
                                -/
    /-
      case right.mk.some
      Γ : Type u_1
      Λ : Type u_2
      inst✝² : Inhabited Λ
      σ : Type u_3
      inst✝¹ : Inhabited σ
      M : Λ → Turing.TM1.Stmt Γ Λ σ
      inst✝ : Fintype σ
      S : Finset Λ
      ss : Turing.TM1.Supports M S
      a : Γ
      q' : Turing.TM1to0.Λ' M
      s : Turing.TM0.Stmt Γ
      v : σ
      q : Turing.TM1.Stmt Γ Λ σ
      h₁ : Membership.mem (Turing.TM1to0.tr M { fst := Option.some q, snd := v } a)  …
      h₂ : Membership.mem ↑(Turing.TM1to0.trStmts M S) { fst := Option.some q, snd : …
      ⊢ Membership.mem (↑(Turing.TM1to0.trStmts M S)) q'
    -/
    cases' q' with q' v'
    /-
      case right.mk.some.mk
      Γ : Type u_1
      Λ : Type u_2
      inst✝² : Inhabited Λ
      σ : Type u_3
      inst✝¹ : Inhabited σ
      M : Λ → Turing.TM1.Stmt Γ Λ σ
      inst✝ : Fintype σ
      S : Finset Λ
      ss : Turing.TM1.Supports M S
      a : Γ
      s : Turing.TM0.Stmt Γ
      v : σ
      q : Turing.TM1.Stmt Γ Λ σ
      h₂ : Membership.mem ↑(Turing.TM1to0.trStmts M S) { fst := Option.some q, snd : …
      q' : Option (Turing.TM1.Stmt Γ Λ σ)
      v' : σ
      h₁ : Membership.mem (Turing.TM1to0.tr M { fst := Option.some q, snd := v } a)  …
      ⊢ Membership.mem ↑(Turing.TM1to0.trStmts M S) { fst := q', snd := v' }
    -/
    simp only [trStmts, Finset.mem_coe] at h₂ ⊢
    /-
      case right.mk.some.mk
      Γ : Type u_1
      Λ : Type u_2
      inst✝² : Inhabited Λ
      σ : Type u_3
      inst✝¹ : Inhabited σ
      M : Λ → Turing.TM1.Stmt Γ Λ σ
      inst✝ : Fintype σ
      S : Finset Λ
      ss : Turing.TM1.Supports M S
      a : Γ
      s : Turing.TM0.Stmt Γ
      v : σ
      q : Turing.TM1.Stmt Γ Λ σ
      q' : Option (Turing.TM1.Stmt Γ Λ σ)
      v' : σ
      h₁ : Membership.mem (Turing.TM1to0.tr M { fst := Option.some q, snd := v } a)  …
      h₂ : Membership.mem (SProd.sprod (Turing.TM1.stmts M S) Finset.univ) { fst :=  …
      ⊢ Membership.mem (SProd.sprod (Turing.TM1.stmts M S) Finset.univ) { fst := q', …
    -/
    rw [Finset.mem_product] at h₂ ⊢
    /-
      case right.mk.some.mk
      Γ : Type u_1
      Λ : Type u_2
      inst✝² : Inhabited Λ
      σ : Type u_3
      inst✝¹ : Inhabited σ
      M : Λ → Turing.TM1.Stmt Γ Λ σ
      inst✝ : Fintype σ
      S : Finset Λ
      ss : Turing.TM1.Supports M S
      a : Γ
      s : Turing.TM0.Stmt Γ
      v : σ
      q : Turing.TM1.Stmt Γ Λ σ
      q' : Option (Turing.TM1.Stmt Γ Λ σ)
      v' : σ
      h₁ : Membership.mem (Turing.TM1to0.tr M { fst := Option.some q, snd := v } a)  …
      h₂ : And (Membership.mem (Turing.TM1.stmts M S) { fst := Option.some q, snd := …
      ⊢ And (Membership.mem (Turing.TM1.stmts M S) { fst := q', snd := v' }.1) (Memb …
    -/
    simp only [Finset.mem_univ, and_true] at h₂ ⊢
    /-
      case right.mk.some.mk
      Γ : Type u_1
      Λ : Type u_2
      inst✝² : Inhabited Λ
      σ : Type u_3
      inst✝¹ : Inhabited σ
      M : Λ → Turing.TM1.Stmt Γ Λ σ
      inst✝ : Fintype σ
      S : Finset Λ
      ss : Turing.TM1.Supports M S
      a : Γ
      s : Turing.TM0.Stmt Γ
      v : σ
      q : Turing.TM1.Stmt Γ Λ σ
      q' : Option (Turing.TM1.Stmt Γ Λ σ)
      v' : σ
      h₁ : Membership.mem (Turing.TM1to0.tr M { fst := Option.some q, snd := v } a)  …
      h₂ : Membership.mem (Turing.TM1.stmts M S) (Option.some q)
      ⊢ Membership.mem (Turing.TM1.stmts M S) q'
    -/
    cases q'; · exact Multiset.mem_cons_self _ _
                /-
                  🎉 no goals
                -/
    /-
      case right.mk.some.mk.some
      Γ : Type u_1
      Λ : Type u_2
      inst✝² : Inhabited Λ
      σ : Type u_3
      inst✝¹ : Inhabited σ
      M : Λ → Turing.TM1.Stmt Γ Λ σ
      inst✝ : Fintype σ
      S : Finset Λ
      ss : Turing.TM1.Supports M S
      a : Γ
      s : Turing.TM0.Stmt Γ
      v : σ
      q : Turing.TM1.Stmt Γ Λ σ
      v' : σ
      h₂ : Membership.mem (Turing.TM1.stmts M S) (Option.some q)
      val✝ : Turing.TM1.Stmt Γ Λ σ
      h₁ : Membership.mem (Turing.TM1to0.tr M { fst := Option.some q, snd := v } a)  …
      ⊢ Membership.mem (Turing.TM1.stmts M S) (Option.some val✝)
    -/
    simp only [tr, Option.mem_def] at h₁
    /-
      case right.mk.some.mk.some
      Γ : Type u_1
      Λ : Type u_2
      inst✝² : Inhabited Λ
      σ : Type u_3
      inst✝¹ : Inhabited σ
      M : Λ → Turing.TM1.Stmt Γ Λ σ
      inst✝ : Fintype σ
      S : Finset Λ
      ss : Turing.TM1.Supports M S
      a : Γ
      s : Turing.TM0.Stmt Γ
      v : σ
      q : Turing.TM1.Stmt Γ Λ σ
      v' : σ
      h₂ : Membership.mem (Turing.TM1.stmts M S) (Option.some q)
      val✝ : Turing.TM1.Stmt Γ Λ σ
      h₁ : Eq (Option.some (Turing.TM1to0.trAux M a q v)) (Option.some { fst := { fs …
      ⊢ Membership.mem (Turing.TM1.stmts M S) (Option.some val✝)
    -/
    have := TM1.stmts_supportsStmt ss h₂
    /-
      case right.mk.some.mk.some
      Γ : Type u_1
      Λ : Type u_2
      inst✝² : Inhabited Λ
      σ : Type u_3
      inst✝¹ : Inhabited σ
      M : Λ → Turing.TM1.Stmt Γ Λ σ
      inst✝ : Fintype σ
      S : Finset Λ
      ss : Turing.TM1.Supports M S
      a : Γ
      s : Turing.TM0.Stmt Γ
      v : σ
      q : Turing.TM1.Stmt Γ Λ σ
      v' : σ
      h₂ : Membership.mem (Turing.TM1.stmts M S) (Option.some q)
      val✝ : Turing.TM1.Stmt Γ Λ σ
      h₁ : Eq (Option.some (Turing.TM1to0.trAux M a q v)) (Option.some { fst := { fs …
      this : Turing.TM1.SupportsStmt S q
      ⊢ Membership.mem (Turing.TM1.stmts M S) (Option.some val✝)
    -/
    revert this; induction q generalizing v with intro hs
    | move d q =>
      cases h₁; refine TM1.stmts_trans ?_ h₂
      unfold TM1.stmts₁
      exact Finset.mem_insert_of_mem TM1.stmts₁_self
    | write b q =>
      cases h₁; refine TM1.stmts_trans ?_ h₂
      unfold TM1.stmts₁
      exact Finset.mem_insert_of_mem TM1.stmts₁_self
    | load b q IH =>
      refine IH _ (TM1.stmts_trans ?_ h₂) h₁ hs
      unfold TM1.stmts₁
      exact Finset.mem_insert_of_mem TM1.stmts₁_self
    | branch p q₁ q₂ IH₁ IH₂ =>
      cases h : p a v <;> rw [trAux, h] at h₁
      · refine IH₂ _ (TM1.stmts_trans ?_ h₂) h₁ hs.2
        unfold TM1.stmts₁
        exact Finset.mem_insert_of_mem (Finset.mem_union_right _ TM1.stmts₁_self)
      · refine IH₁ _ (TM1.stmts_trans ?_ h₂) h₁ hs.1
        unfold TM1.stmts₁
        exact Finset.mem_insert_of_mem (Finset.mem_union_left _ TM1.stmts₁_self)
    | goto l =>
      cases h₁
      exact Finset.some_mem_insertNone.2 (Finset.mem_biUnion.2 ⟨_, hs _ _, TM1.stmts₁_self⟩)
    | halt => cases h₁


theorem exists_enc_dec [Inhabited Γ] [Finite Γ] :
    ∃ (n : ℕ) (enc : Γ → List.Vector Bool n) (dec : List.Vector Bool n → Γ),
      enc default = Vector.replicate n false ∧ ∀ a, dec (enc a) = a := by
  /-
    Γ : Type u_1
    inst✝¹ : Inhabited Γ
    inst✝ : Finite Γ
    ⊢ Exists fun n => Exists fun enc => Exists fun dec => And (Eq (enc Inhabited.d …
  -/
  rcases Finite.exists_equiv_fin Γ with ⟨n, ⟨e⟩⟩
  /-
    case intro.intro
    Γ : Type u_1
    inst✝¹ : Inhabited Γ
    inst✝ : Finite Γ
    n : Nat
    e : Equiv Γ (Fin n)
    ⊢ Exists fun n => Exists fun enc => Exists fun dec => And (Eq (enc Inhabited.d …
  -/
  letI : DecidableEq Γ := e.decidableEq
  let G : Fin n ↪ Fin n → Bool :=
    ⟨fun a b ↦ a = b, fun a b h ↦
      Bool.of_decide_true <| (congr_fun h b).trans <| Bool.decide_true rfl⟩
  /-
    case intro.intro
    Γ : Type u_1
    inst✝¹ : Inhabited Γ
    inst✝ : Finite Γ
    n : Nat
    e : Equiv Γ (Fin n)
    this : DecidableEq Γ := e.decidableEq
    G : Function.Embedding (Fin n) (Fin n → Bool) := { toFun := fun a b => Decidab …
    ⊢ Exists fun n => Exists fun enc => Exists fun dec => And (Eq (enc Inhabited.d …
  -/
  let H := (e.toEmbedding.trans G).trans (Equiv.vectorEquivFin _ _).symm.toEmbedding
  /-
    case intro.intro
    Γ : Type u_1
    inst✝¹ : Inhabited Γ
    inst✝ : Finite Γ
    n : Nat
    e : Equiv Γ (Fin n)
    this : DecidableEq Γ := e.decidableEq
    G : Function.Embedding (Fin n) (Fin n → Bool) := { toFun := fun a b => Decidab …
    H : Function.Embedding Γ (List.Vector Bool n) := (e.toEmbedding.trans G).trans …
    ⊢ Exists fun n => Exists fun enc => Exists fun dec => And (Eq (enc Inhabited.d …
  -/
  let enc := H.setValue default (Vector.replicate n false)
  /-
    case intro.intro
    Γ : Type u_1
    inst✝¹ : Inhabited Γ
    inst✝ : Finite Γ
    n : Nat
    e : Equiv Γ (Fin n)
    this : DecidableEq Γ := e.decidableEq
    G : Function.Embedding (Fin n) (Fin n → Bool) := { toFun := fun a b => Decidab …
    H : Function.Embedding Γ (List.Vector Bool n) := (e.toEmbedding.trans G).trans …
    enc : Function.Embedding Γ (List.Vector Bool n) := H.setValue Inhabited.defaul …
    ⊢ Exists fun n => Exists fun enc => Exists fun dec => And (Eq (enc Inhabited.d …
  -/
  exact ⟨_, enc, Function.invFun enc, H.setValue_eq _ _, Function.leftInverse_invFun enc.2⟩
  /-
    🎉 no goals
  -/


local notation "Stmt₁" => Stmt Γ Λ σ


local notation "Cfg₁" => Cfg Γ Λ σ


/-- The configuration state of the TM. -/
inductive Λ'
  | normal : Λ → Λ'
  | write : Γ → Stmt₁ → Λ'


local notation "Λ'₁" => @Λ' Γ Λ σ  -- Porting note (https://github.com/leanprover-community/mathlib4/issues/10750): added this to clean up types.


instance [Inhabited Λ] : Inhabited Λ'₁ :=
  ⟨Λ'.normal default⟩


local notation "Stmt'₁" => Stmt Bool Λ'₁ σ


local notation "Cfg'₁" => Cfg Bool Λ'₁ σ


/-- Read a vector of length `n` from the tape. -/
def readAux : ∀ n, (List.Vector Bool n → Stmt'₁) → Stmt'₁
  | 0, f => f Vector.nil
  | i + 1, f =>
    Stmt.branch (fun a _ ↦ a) (Stmt.move Dir.right <| readAux i fun v ↦ f (true ::ᵥ v))
      (Stmt.move Dir.right <| readAux i fun v ↦ f (false ::ᵥ v))


/-- A move left or right corresponds to `n` moves across the super-cell. -/
def move (d : Dir) (q : Stmt'₁) : Stmt'₁ :=
  (Stmt.move d)^[n] q


local notation "moveₙ" => @move Γ Λ σ n  -- Porting note (https://github.com/leanprover-community/mathlib4/issues/10750): added this to clean up types.


/-- To read a symbol from the tape, we use `readAux` to traverse the symbol,
then return to the original position with `n` moves to the left. -/
def read (f : Γ → Stmt'₁) : Stmt'₁ :=
  readAux n fun v ↦ moveₙ Dir.left <| f (dec v)


/-- Write a list of bools on the tape. -/
def write : List Bool → Stmt'₁ → Stmt'₁
  | [], q => q
  | a :: l, q => (Stmt.write fun _ _ ↦ a) <| Stmt.move Dir.right <| write l q


/-- Translate a normal instruction. For the `write` command, we use a `goto` indirection so that
we can access the current value of the tape. -/
def trNormal : Stmt₁ → Stmt'₁
  | Stmt.move d q => moveₙ d <| trNormal q
  | Stmt.write f q => read dec fun a ↦ Stmt.goto fun _ s ↦ Λ'.write (f a s) q
  | Stmt.load f q => read dec fun a ↦ (Stmt.load fun _ s ↦ f a s) <| trNormal q
  | Stmt.branch p q₁ q₂ =>
    read dec fun a ↦ Stmt.branch (fun _ s ↦ p a s) (trNormal q₁) (trNormal q₂)
  | Stmt.goto l => read dec fun a ↦ Stmt.goto fun _ s ↦ Λ'.normal (l a s)
  | Stmt.halt => Stmt.halt


theorem stepAux_move (d : Dir) (q : Stmt'₁) (v : σ) (T : Tape Bool) :
    stepAux (moveₙ d q) v T = stepAux q v ((Tape.move d)^[n] T) := by
  /-
    Γ : Type u_1
    Λ : Type u_2
    σ : Type u_3
    n : Nat
    d : Turing.Dir
    q : Turing.TM1.Stmt Bool Turing.TM1to1.Λ' σ
    v : σ
    T : Turing.Tape Bool
    ⊢ Eq (Turing.TM1.stepAux (Turing.TM1to1.move d q) v T) (Turing.TM1.stepAux q v …
  -/
  suffices ∀ i, stepAux ((Stmt.move d)^[i] q) v T = stepAux q v ((Tape.move d)^[i] T) from this n
  /-
    Γ : Type u_1
    Λ : Type u_2
    σ : Type u_3
    n : Nat
    d : Turing.Dir
    q : Turing.TM1.Stmt Bool Turing.TM1to1.Λ' σ
    v : σ
    T : Turing.Tape Bool
    ⊢ ∀ (i : Nat), Eq (Turing.TM1.stepAux (Nat.iterate (Turing.TM1.Stmt.move d) i  …
  -/
  intro i; induction' i with i IH generalizing T; · rfl
                                                    /-
                                                      🎉 no goals
                                                    -/
  /-
    case succ
    Γ : Type u_1
    Λ : Type u_2
    σ : Type u_3
    n : Nat
    d : Turing.Dir
    q : Turing.TM1.Stmt Bool Turing.TM1to1.Λ' σ
    v : σ
    i : Nat
    IH : ∀ (T : Turing.Tape Bool), Eq (Turing.TM1.stepAux (Nat.iterate (Turing.TM1 …
    T : Turing.Tape Bool
    ⊢ Eq (Turing.TM1.stepAux (Nat.iterate (Turing.TM1.Stmt.move d) (HAdd.hAdd i 1) …
  -/
  rw [iterate_succ', iterate_succ]
  /-
    case succ
    Γ : Type u_1
    Λ : Type u_2
    σ : Type u_3
    n : Nat
    d : Turing.Dir
    q : Turing.TM1.Stmt Bool Turing.TM1to1.Λ' σ
    v : σ
    i : Nat
    IH : ∀ (T : Turing.Tape Bool), Eq (Turing.TM1.stepAux (Nat.iterate (Turing.TM1 …
    T : Turing.Tape Bool
    ⊢ Eq (Turing.TM1.stepAux (Function.comp (Turing.TM1.Stmt.move d) (Nat.iterate  …
  -/
  simp only [stepAux, Function.comp_apply]
  /-
    case succ
    Γ : Type u_1
    Λ : Type u_2
    σ : Type u_3
    n : Nat
    d : Turing.Dir
    q : Turing.TM1.Stmt Bool Turing.TM1to1.Λ' σ
    v : σ
    i : Nat
    IH : ∀ (T : Turing.Tape Bool), Eq (Turing.TM1.stepAux (Nat.iterate (Turing.TM1 …
    T : Turing.Tape Bool
    ⊢ Eq (Turing.TM1.stepAux (Nat.iterate (Turing.TM1.Stmt.move d) i q) v (Turing. …
  -/
  rw [IH]
  /-
    🎉 no goals
  -/


theorem supportsStmt_move {S : Finset Λ'₁} {d : Dir} {q : Stmt'₁} :
    SupportsStmt S (moveₙ d q) = SupportsStmt S q := by
  /-
    Γ : Type u_1
    Λ : Type u_2
    σ : Type u_3
    n : Nat
    S : Finset Turing.TM1to1.Λ'
    d : Turing.Dir
    q : Turing.TM1.Stmt Bool Turing.TM1to1.Λ' σ
    ⊢ Eq (Turing.TM1.SupportsStmt S (Turing.TM1to1.move d q)) (Turing.TM1.Supports …
  -/
  suffices ∀ {i}, SupportsStmt S ((Stmt.move d)^[i] q) = _ from this
  /-
    Γ : Type u_1
    Λ : Type u_2
    σ : Type u_3
    n : Nat
    S : Finset Turing.TM1to1.Λ'
    d : Turing.Dir
    q : Turing.TM1.Stmt Bool Turing.TM1to1.Λ' σ
    ⊢ ∀ {i : Nat}, Eq (Turing.TM1.SupportsStmt S (Nat.iterate (Turing.TM1.Stmt.mov …
  -/
                                          /-
                                            🎉 no goals
                                          -/
  intro i; induction i generalizing q <;> simp only [*, iterate]; rfl
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


theorem supportsStmt_write {S : Finset Λ'₁} {l : List Bool} {q : Stmt'₁} :
    SupportsStmt S (write l q) = SupportsStmt S q := by
  /-
    Γ : Type u_1
    Λ : Type u_2
    σ : Type u_3
    S : Finset Turing.TM1to1.Λ'
    l : List Bool
    q : Turing.TM1.Stmt Bool Turing.TM1to1.Λ' σ
    ⊢ Eq (Turing.TM1.SupportsStmt S (Turing.TM1to1.write l q)) (Turing.TM1.Support …
  -/
                               /-
                                 🎉 no goals
                               -/
  induction' l with _ l IH <;> simp only [write, SupportsStmt, *]
                               /-
                                 🎉 no goals
                               -/


theorem supportsStmt_read {S : Finset Λ'₁} :
    ∀ {f : Γ → Stmt'₁}, (∀ a, SupportsStmt S (f a)) → SupportsStmt S (read dec f) :=
  suffices
    ∀ (i) (f : List.Vector Bool i → Stmt'₁),
      (∀ v, SupportsStmt S (f v)) → SupportsStmt S (readAux i f)
                               /-
                                 Γ : Type u_1
                                 Λ : Type u_2
                                 σ : Type u_3
                                 n : Nat
                                 dec : List.Vector Bool n → Γ
                                 S : Finset Turing.TM1to1.Λ'
                                 f✝ : Γ → Turing.TM1.Stmt Bool Turing.TM1to1.Λ' σ
                                 this : ∀ (i : Nat) (f : List.Vector Bool i → Turing.TM1.Stmt Bool Turing.TM1to …
                                 hf : ∀ (a : Γ), Turing.TM1.SupportsStmt S (f✝ a)
                                 ⊢ ∀ (v : List.Vector Bool n), Turing.TM1.SupportsStmt S (Turing.TM1to1.move Tu …
                               -/
    from fun hf ↦ this n _ (by intro; simp only [supportsStmt_move, hf])
  /-
    Γ : Type u_1
    Λ : Type u_2
    σ : Type u_3
    n : Nat
    dec : List.Vector Bool n → Γ
    S : Finset Turing.TM1to1.Λ'
    f✝ : Γ → Turing.TM1.Stmt Bool Turing.TM1to1.Λ' σ
    i : Nat
    f : List.Vector Bool i → Turing.TM1.Stmt Bool Turing.TM1to1.Λ' σ
    hf : ∀ (v : List.Vector Bool i), Turing.TM1.SupportsStmt S (f v)
    ⊢ Turing.TM1.SupportsStmt S (Turing.TM1to1.readAux i f)
  -/
                                      /-
                                        🎉 no goals
                                      -/
                            /-
                              🎉 no goals
                            -/
  /-
    case succ
    Γ : Type u_1
    Λ : Type u_2
    σ : Type u_3
    n : Nat
    dec : List.Vector Bool n → Γ
    S : Finset Turing.TM1to1.Λ'
    f✝ : Γ → Turing.TM1.Stmt Bool Turing.TM1to1.Λ' σ
    i : Nat
    IH : ∀ (f : List.Vector Bool i → Turing.TM1.Stmt Bool Turing.TM1to1.Λ' σ), (∀  …
    f : List.Vector Bool (HAdd.hAdd i 1) → Turing.TM1.Stmt Bool Turing.TM1to1.Λ' σ
    hf : ∀ (v : List.Vector Bool (HAdd.hAdd i 1)), Turing.TM1.SupportsStmt S (f v)
    ⊢ Turing.TM1.SupportsStmt S (Turing.TM1to1.readAux (HAdd.hAdd i 1) f)
  -/
                                         /-
                                           🎉 no goals
                                         -/
  fun i f hf ↦ by
                                         /-
                                           🎉 no goals
                                         -/
  induction' i with i IH; · exact hf _
  constructor <;> apply IH <;> intro <;> apply hf


/-- The low level tape corresponding to the given tape over alphabet `Γ`. -/
def trTape' (L R : ListBlank Γ) : Tape Bool := by
  refine
      Tape.mk' (L.flatMap (fun x ↦ (enc x).toList.reverse) ⟨n, ?_⟩)
        (R.flatMap (fun x ↦ (enc x).toList) ⟨n, ?_⟩) <;>
    /-
      case refine_1
      Γ : Type u_1
      Λ : Type u_2
      σ : Type u_3
      n : Nat
      enc : Γ → List.Vector Bool n
      dec : List.Vector Bool n → Γ
      M : Λ → Turing.TM1.Stmt Γ Λ σ
      inst✝ : Inhabited Γ
      enc0 : Eq (enc Inhabited.default) (List.Vector.replicate n Bool.false)
      L R : Turing.ListBlank Γ
      ⊢ Eq ((fun x => (enc x).toList.reverse) Inhabited.default) (List.replicate n I …
    -/
    /-
      🎉 no goals
    -/
    simp only [enc0, Vector.replicate, List.reverse_replicate, Bool.default_bool, Vector.toList_mk]
    /-
      🎉 no goals
    -/


/-- The low level tape corresponding to the given tape over alphabet `Γ`. -/
def trTape (T : Tape Γ) : Tape Bool :=
  trTape' enc0 T.left T.right₀


theorem trTape_mk' (L R : ListBlank Γ) : trTape enc0 (Tape.mk' L R) = trTape' enc0 L R := by
  /-
    Γ : Type u_1
    n : Nat
    enc : Γ → List.Vector Bool n
    inst✝ : Inhabited Γ
    enc0 : Eq (enc Inhabited.default) (List.Vector.replicate n Bool.false)
    L R : Turing.ListBlank Γ
    ⊢ Eq (Turing.TM1to1.trTape enc0 (Turing.Tape.mk' L R)) (Turing.TM1to1.trTape'  …
  -/
  simp only [trTape, Tape.mk'_left, Tape.mk'_right₀]
  /-
    🎉 no goals
  -/


/-- The top level program. -/
def tr : Λ'₁ → Stmt'₁
  | Λ'.normal l => trNormal dec (M l)
  | Λ'.write a q => write (enc a).toList <| moveₙ Dir.left <| trNormal dec q


/-- The machine configuration translation. -/
def trCfg : Cfg₁ → Cfg'₁
  | ⟨l, v, T⟩ => ⟨l.map Λ'.normal, v, trTape enc0 T⟩


theorem trTape'_move_left (L R : ListBlank Γ) :
    (Tape.move Dir.left)^[n] (trTape' enc0 L R) = trTape' enc0 L.tail (R.cons L.head) := by
  /-
    Γ : Type u_1
    n : Nat
    enc : Γ → List.Vector Bool n
    inst✝ : Inhabited Γ
    enc0 : Eq (enc Inhabited.default) (List.Vector.replicate n Bool.false)
    L R : Turing.ListBlank Γ
    ⊢ Eq (Nat.iterate (Turing.Tape.move Turing.Dir.left) n (Turing.TM1to1.trTape'  …
  -/
  obtain ⟨a, L, rfl⟩ := L.exists_cons
  /-
    case intro.intro
    Γ : Type u_1
    n : Nat
    enc : Γ → List.Vector Bool n
    inst✝ : Inhabited Γ
    enc0 : Eq (enc Inhabited.default) (List.Vector.replicate n Bool.false)
    R : Turing.ListBlank Γ
    a : Γ
    L : Turing.ListBlank Γ
    ⊢ Eq (Nat.iterate (Turing.Tape.move Turing.Dir.left) n (Turing.TM1to1.trTape'  …
  -/
  simp only [trTape', ListBlank.cons_flatMap, ListBlank.head_cons, ListBlank.tail_cons]
  suffices ∀ {L' R' l₁ l₂} (_ : Vector.toList (enc a) = List.reverseAux l₁ l₂),
      (Tape.move Dir.left)^[l₁.length]
      (Tape.mk' (ListBlank.append l₁ L') (ListBlank.append l₂ R')) =
      Tape.mk' L' (ListBlank.append (Vector.toList (enc a)) R') by
    simpa only [List.length_reverse, Vector.toList_length] using this (List.reverse_reverse _).symm
  /-
    case intro.intro
    Γ : Type u_1
    n : Nat
    enc : Γ → List.Vector Bool n
    inst✝ : Inhabited Γ
    enc0 : Eq (enc Inhabited.default) (List.Vector.replicate n Bool.false)
    R : Turing.ListBlank Γ
    a : Γ
    L : Turing.ListBlank Γ
    ⊢ ∀ {L' R' : Turing.ListBlank Bool} {l₁ l₂ : List Bool}, Eq (enc a).toList (l₁ …
  -/
  intro _ _ l₁ l₂ e
  /-
    case intro.intro
    Γ : Type u_1
    n : Nat
    enc : Γ → List.Vector Bool n
    inst✝ : Inhabited Γ
    enc0 : Eq (enc Inhabited.default) (List.Vector.replicate n Bool.false)
    R : Turing.ListBlank Γ
    a : Γ
    L : Turing.ListBlank Γ
    L'✝ R'✝ : Turing.ListBlank Bool
    l₁ l₂ : List Bool
    e : Eq (enc a).toList (l₁.reverseAux l₂)
    ⊢ Eq (Nat.iterate (Turing.Tape.move Turing.Dir.left) l₁.length (Turing.Tape.mk …
  -/
  induction' l₁ with b l₁ IH generalizing l₂
    /-
      case intro.intro.nil
      Γ : Type u_1
      n : Nat
      enc : Γ → List.Vector Bool n
      inst✝ : Inhabited Γ
      enc0 : Eq (enc Inhabited.default) (List.Vector.replicate n Bool.false)
      R : Turing.ListBlank Γ
      a : Γ
      L : Turing.ListBlank Γ
      L'✝ R'✝ : Turing.ListBlank Bool
      l₂ : List Bool
      e : Eq (enc a).toList (List.nil.reverseAux l₂)
      ⊢ Eq (Nat.iterate (Turing.Tape.move Turing.Dir.left) List.nil.length (Turing.T …
    -/
  · cases e
    /-
      case intro.intro.nil.refl
      Γ : Type u_1
      n : Nat
      enc : Γ → List.Vector Bool n
      inst✝ : Inhabited Γ
      enc0 : Eq (enc Inhabited.default) (List.Vector.replicate n Bool.false)
      R : Turing.ListBlank Γ
      a : Γ
      L : Turing.ListBlank Γ
      L'✝ R'✝ : Turing.ListBlank Bool
      ⊢ Eq (Nat.iterate (Turing.Tape.move Turing.Dir.left) List.nil.length (Turing.T …
    -/
    rfl
    /-
      🎉 no goals
    -/
  /-
    case intro.intro.cons
    Γ : Type u_1
    n : Nat
    enc : Γ → List.Vector Bool n
    inst✝ : Inhabited Γ
    enc0 : Eq (enc Inhabited.default) (List.Vector.replicate n Bool.false)
    R : Turing.ListBlank Γ
    a : Γ
    L : Turing.ListBlank Γ
    L'✝ R'✝ : Turing.ListBlank Bool
    b : Bool
    l₁ : List Bool
    IH : ∀ {l₂ : List Bool}, Eq (enc a).toList (l₁.reverseAux l₂) → Eq (Nat.iterat …
    l₂ : List Bool
    e : Eq (enc a).toList ((List.cons b l₁).reverseAux l₂)
    ⊢ Eq (Nat.iterate (Turing.Tape.move Turing.Dir.left) (List.cons b l₁).length ( …
  -/
  simp only [List.length, List.cons_append, iterate_succ_apply]
  /-
    case intro.intro.cons
    Γ : Type u_1
    n : Nat
    enc : Γ → List.Vector Bool n
    inst✝ : Inhabited Γ
    enc0 : Eq (enc Inhabited.default) (List.Vector.replicate n Bool.false)
    R : Turing.ListBlank Γ
    a : Γ
    L : Turing.ListBlank Γ
    L'✝ R'✝ : Turing.ListBlank Bool
    b : Bool
    l₁ : List Bool
    IH : ∀ {l₂ : List Bool}, Eq (enc a).toList (l₁.reverseAux l₂) → Eq (Nat.iterat …
    l₂ : List Bool
    e : Eq (enc a).toList ((List.cons b l₁).reverseAux l₂)
    ⊢ Eq (Nat.iterate (Turing.Tape.move Turing.Dir.left) l₁.length (Turing.Tape.mo …
  -/
  convert IH e
  /-
    case h.e'_2.h.e'_4
    Γ : Type u_1
    n : Nat
    enc : Γ → List.Vector Bool n
    inst✝ : Inhabited Γ
    enc0 : Eq (enc Inhabited.default) (List.Vector.replicate n Bool.false)
    R : Turing.ListBlank Γ
    a : Γ
    L : Turing.ListBlank Γ
    L'✝ R'✝ : Turing.ListBlank Bool
    b : Bool
    l₁ : List Bool
    IH : ∀ {l₂ : List Bool}, Eq (enc a).toList (l₁.reverseAux l₂) → Eq (Nat.iterat …
    l₂ : List Bool
    e : Eq (enc a).toList ((List.cons b l₁).reverseAux l₂)
    ⊢ Eq (Turing.Tape.move Turing.Dir.left (Turing.Tape.mk' (Turing.ListBlank.appe …
  -/
  simp only [ListBlank.tail_cons, ListBlank.append, Tape.move_left_mk', ListBlank.head_cons]
  /-
    🎉 no goals
  -/


theorem trTape'_move_right (L R : ListBlank Γ) :
    (Tape.move Dir.right)^[n] (trTape' enc0 L R) = trTape' enc0 (L.cons R.head) R.tail := by
  suffices ∀ i L, (Tape.move Dir.right)^[i] ((Tape.move Dir.left)^[i] L) = L by
    refine (Eq.symm ?_).trans (this n _)
    simp only [trTape'_move_left, ListBlank.cons_head_tail, ListBlank.head_cons,
      ListBlank.tail_cons]
  /-
    Γ : Type u_1
    n : Nat
    enc : Γ → List.Vector Bool n
    inst✝ : Inhabited Γ
    enc0 : Eq (enc Inhabited.default) (List.Vector.replicate n Bool.false)
    L R : Turing.ListBlank Γ
    ⊢ ∀ (i : Nat) (L : Turing.Tape Bool), Eq (Nat.iterate (Turing.Tape.move Turing …
  -/
  intro i _
  /-
    Γ : Type u_1
    n : Nat
    enc : Γ → List.Vector Bool n
    inst✝ : Inhabited Γ
    enc0 : Eq (enc Inhabited.default) (List.Vector.replicate n Bool.false)
    L R : Turing.ListBlank Γ
    i : Nat
    L✝ : Turing.Tape Bool
    ⊢ Eq (Nat.iterate (Turing.Tape.move Turing.Dir.right) i (Nat.iterate (Turing.T …
  -/
  induction' i with i IH
    /-
      case zero
      Γ : Type u_1
      n : Nat
      enc : Γ → List.Vector Bool n
      inst✝ : Inhabited Γ
      enc0 : Eq (enc Inhabited.default) (List.Vector.replicate n Bool.false)
      L R : Turing.ListBlank Γ
      L✝ : Turing.Tape Bool
      ⊢ Eq (Nat.iterate (Turing.Tape.move Turing.Dir.right) 0 (Nat.iterate (Turing.T …
    -/
  · rfl
    /-
      🎉 no goals
    -/
  /-
    case succ
    Γ : Type u_1
    n : Nat
    enc : Γ → List.Vector Bool n
    inst✝ : Inhabited Γ
    enc0 : Eq (enc Inhabited.default) (List.Vector.replicate n Bool.false)
    L R : Turing.ListBlank Γ
    L✝ : Turing.Tape Bool
    i : Nat
    IH : Eq (Nat.iterate (Turing.Tape.move Turing.Dir.right) i (Nat.iterate (Turin …
    ⊢ Eq (Nat.iterate (Turing.Tape.move Turing.Dir.right) (HAdd.hAdd i 1) (Nat.ite …
  -/
  rw [iterate_succ_apply, iterate_succ_apply', Tape.move_left_right, IH]
  /-
    🎉 no goals
  -/


theorem stepAux_write (q : Stmt'₁) (v : σ) (a b : Γ) (L R : ListBlank Γ) :
    stepAux (write (enc a).toList q) v (trTape' enc0 L (ListBlank.cons b R)) =
      stepAux q v (trTape' enc0 (ListBlank.cons a L) R) := by
  /-
    Γ : Type u_1
    Λ : Type u_2
    σ : Type u_3
    n : Nat
    enc : Γ → List.Vector Bool n
    inst✝ : Inhabited Γ
    enc0 : Eq (enc Inhabited.default) (List.Vector.replicate n Bool.false)
    q : Turing.TM1.Stmt Bool Turing.TM1to1.Λ' σ
    v : σ
    a b : Γ
    L R : Turing.ListBlank Γ
    ⊢ Eq (Turing.TM1.stepAux (Turing.TM1to1.write (enc a).toList q) v (Turing.TM1t …
  -/
  simp only [trTape', ListBlank.cons_flatMap]
  suffices ∀ {L' R'} (l₁ l₂ l₂' : List Bool) (_ : l₂'.length = l₂.length),
      stepAux (write l₂ q) v (Tape.mk' (ListBlank.append l₁ L') (ListBlank.append l₂' R')) =
      stepAux q v (Tape.mk' (L'.append (List.reverseAux l₂ l₁)) R') by
    exact this [] _ _ ((enc b).2.trans (enc a).2.symm)
  /-
    Γ : Type u_1
    Λ : Type u_2
    σ : Type u_3
    n : Nat
    enc : Γ → List.Vector Bool n
    inst✝ : Inhabited Γ
    enc0 : Eq (enc Inhabited.default) (List.Vector.replicate n Bool.false)
    q : Turing.TM1.Stmt Bool Turing.TM1to1.Λ' σ
    v : σ
    a b : Γ
    L R : Turing.ListBlank Γ
    ⊢ ∀ {L' R' : Turing.ListBlank Bool} (l₁ l₂ l₂' : List Bool), Eq l₂'.length l₂. …
  -/
  clear a b L R
  /-
    Γ : Type u_1
    Λ : Type u_2
    σ : Type u_3
    n : Nat
    enc : Γ → List.Vector Bool n
    inst✝ : Inhabited Γ
    enc0 : Eq (enc Inhabited.default) (List.Vector.replicate n Bool.false)
    q : Turing.TM1.Stmt Bool Turing.TM1to1.Λ' σ
    v : σ
    ⊢ ∀ {L' R' : Turing.ListBlank Bool} (l₁ l₂ l₂' : List Bool), Eq l₂'.length l₂. …
  -/
  intro L' R' l₁ l₂ l₂' e
  /-
    Γ : Type u_1
    Λ : Type u_2
    σ : Type u_3
    n : Nat
    enc : Γ → List.Vector Bool n
    inst✝ : Inhabited Γ
    enc0 : Eq (enc Inhabited.default) (List.Vector.replicate n Bool.false)
    q : Turing.TM1.Stmt Bool Turing.TM1to1.Λ' σ
    v : σ
    L' R' : Turing.ListBlank Bool
    l₁ l₂ l₂' : List Bool
    e : Eq l₂'.length l₂.length
    ⊢ Eq (Turing.TM1.stepAux (Turing.TM1to1.write l₂ q) v (Turing.Tape.mk' (Turing …
  -/
  induction' l₂ with a l₂ IH generalizing l₁ l₂'
    /-
      case nil
      Γ : Type u_1
      Λ : Type u_2
      σ : Type u_3
      n : Nat
      enc : Γ → List.Vector Bool n
      inst✝ : Inhabited Γ
      enc0 : Eq (enc Inhabited.default) (List.Vector.replicate n Bool.false)
      q : Turing.TM1.Stmt Bool Turing.TM1to1.Λ' σ
      v : σ
      L' R' : Turing.ListBlank Bool
      l₁ l₂' : List Bool
      e : Eq l₂'.length List.nil.length
      ⊢ Eq (Turing.TM1.stepAux (Turing.TM1to1.write List.nil q) v (Turing.Tape.mk' ( …
    -/
  · cases List.length_eq_zero.1 e
    /-
      case nil.refl
      Γ : Type u_1
      Λ : Type u_2
      σ : Type u_3
      n : Nat
      enc : Γ → List.Vector Bool n
      inst✝ : Inhabited Γ
      enc0 : Eq (enc Inhabited.default) (List.Vector.replicate n Bool.false)
      q : Turing.TM1.Stmt Bool Turing.TM1to1.Λ' σ
      v : σ
      L' R' : Turing.ListBlank Bool
      l₁ : List Bool
      e : Eq List.nil.length List.nil.length
      ⊢ Eq (Turing.TM1.stepAux (Turing.TM1to1.write List.nil q) v (Turing.Tape.mk' ( …
    -/
    rfl
    /-
      🎉 no goals
    -/
  /-
    case cons
    Γ : Type u_1
    Λ : Type u_2
    σ : Type u_3
    n : Nat
    enc : Γ → List.Vector Bool n
    inst✝ : Inhabited Γ
    enc0 : Eq (enc Inhabited.default) (List.Vector.replicate n Bool.false)
    q : Turing.TM1.Stmt Bool Turing.TM1to1.Λ' σ
    v : σ
    L' R' : Turing.ListBlank Bool
    a : Bool
    l₂ : List Bool
    IH : ∀ (l₁ l₂' : List Bool), Eq l₂'.length l₂.length → Eq (Turing.TM1.stepAux  …
    l₁ l₂' : List Bool
    e : Eq l₂'.length (List.cons a l₂).length
    ⊢ Eq (Turing.TM1.stepAux (Turing.TM1to1.write (List.cons a l₂) q) v (Turing.Ta …
  -/
  cases' l₂' with b l₂' <;>
    /-
      case cons.nil
      Γ : Type u_1
      Λ : Type u_2
      σ : Type u_3
      n : Nat
      enc : Γ → List.Vector Bool n
      inst✝ : Inhabited Γ
      enc0 : Eq (enc Inhabited.default) (List.Vector.replicate n Bool.false)
      q : Turing.TM1.Stmt Bool Turing.TM1to1.Λ' σ
      v : σ
      L' R' : Turing.ListBlank Bool
      a : Bool
      l₂ : List Bool
      IH : ∀ (l₁ l₂' : List Bool), Eq l₂'.length l₂.length → Eq (Turing.TM1.stepAux  …
      l₁ : List Bool
      e : Eq List.nil.length (List.cons a l₂).length
      ⊢ Eq (Turing.TM1.stepAux (Turing.TM1to1.write (List.cons a l₂) q) v (Turing.Ta …
    -/
    /-
      🎉 no goals
    -/
    simp only [List.length_nil, List.length_cons, Nat.succ_inj', reduceCtorEq] at e
  /-
    case cons.cons
    Γ : Type u_1
    Λ : Type u_2
    σ : Type u_3
    n : Nat
    enc : Γ → List.Vector Bool n
    inst✝ : Inhabited Γ
    enc0 : Eq (enc Inhabited.default) (List.Vector.replicate n Bool.false)
    q : Turing.TM1.Stmt Bool Turing.TM1to1.Λ' σ
    v : σ
    L' R' : Turing.ListBlank Bool
    a : Bool
    l₂ : List Bool
    IH : ∀ (l₁ l₂' : List Bool), Eq l₂'.length l₂.length → Eq (Turing.TM1.stepAux  …
    l₁ : List Bool
    b : Bool
    l₂' : List Bool
    e : Eq l₂'.length l₂.length
    ⊢ Eq (Turing.TM1.stepAux (Turing.TM1to1.write (List.cons a l₂) q) v (Turing.Ta …
  -/
  rw [List.reverseAux, ← IH (a :: l₁) l₂' e]
  /-
    case cons.cons
    Γ : Type u_1
    Λ : Type u_2
    σ : Type u_3
    n : Nat
    enc : Γ → List.Vector Bool n
    inst✝ : Inhabited Γ
    enc0 : Eq (enc Inhabited.default) (List.Vector.replicate n Bool.false)
    q : Turing.TM1.Stmt Bool Turing.TM1to1.Λ' σ
    v : σ
    L' R' : Turing.ListBlank Bool
    a : Bool
    l₂ : List Bool
    IH : ∀ (l₁ l₂' : List Bool), Eq l₂'.length l₂.length → Eq (Turing.TM1.stepAux  …
    l₁ : List Bool
    b : Bool
    l₂' : List Bool
    e : Eq l₂'.length l₂.length
    ⊢ Eq (Turing.TM1.stepAux (Turing.TM1to1.write (List.cons a l₂) q) v (Turing.Ta …
  -/
  simp [stepAux, ListBlank.append, write]
  /-
    🎉 no goals
  -/


theorem stepAux_read (f : Γ → Stmt'₁) (v : σ) (L R : ListBlank Γ) :
    stepAux (read dec f) v (trTape' enc0 L R) = stepAux (f R.head) v (trTape' enc0 L R) := by
  suffices ∀ f, stepAux (readAux n f) v (trTape' enc0 L R) =
      stepAux (f (enc R.head)) v (trTape' enc0 (L.cons R.head) R.tail) by
    rw [read, this, stepAux_move, encdec, trTape'_move_left enc0]
    simp only [ListBlank.head_cons, ListBlank.cons_head_tail, ListBlank.tail_cons]
  /-
    Γ : Type u_1
    Λ : Type u_2
    σ : Type u_3
    n : Nat
    enc : Γ → List.Vector Bool n
    dec : List.Vector Bool n → Γ
    inst✝ : Inhabited Γ
    enc0 : Eq (enc Inhabited.default) (List.Vector.replicate n Bool.false)
    encdec : ∀ (a : Γ), Eq (dec (enc a)) a
    f : Γ → Turing.TM1.Stmt Bool Turing.TM1to1.Λ' σ
    v : σ
    L R : Turing.ListBlank Γ
    ⊢ ∀ (f : List.Vector Bool n → Turing.TM1.Stmt Bool Turing.TM1to1.Λ' σ), Eq (Tu …
  -/
  obtain ⟨a, R, rfl⟩ := R.exists_cons
  simp only [ListBlank.head_cons, ListBlank.tail_cons, trTape', ListBlank.cons_flatMap,
    ListBlank.append_assoc]
  suffices ∀ i f L' R' l₁ l₂ h,
      stepAux (readAux i f) v (Tape.mk' (ListBlank.append l₁ L') (ListBlank.append l₂ R')) =
      stepAux (f ⟨l₂, h⟩) v (Tape.mk' (ListBlank.append (l₂.reverseAux l₁) L') R') by
    intro f
    -- Porting note: Here was `change`.
    exact this n f (L.flatMap (fun x => (enc x).1.reverse) _)
      (R.flatMap (fun x => (enc x).1) _) [] _ (enc a).2
  /-
    case intro.intro
    Γ : Type u_1
    Λ : Type u_2
    σ : Type u_3
    n : Nat
    enc : Γ → List.Vector Bool n
    dec : List.Vector Bool n → Γ
    inst✝ : Inhabited Γ
    enc0 : Eq (enc Inhabited.default) (List.Vector.replicate n Bool.false)
    encdec : ∀ (a : Γ), Eq (dec (enc a)) a
    f : Γ → Turing.TM1.Stmt Bool Turing.TM1to1.Λ' σ
    v : σ
    L : Turing.ListBlank Γ
    a : Γ
    R : Turing.ListBlank Γ
    ⊢ ∀ (i : Nat) (f : List.Vector Bool i → Turing.TM1.Stmt Bool Turing.TM1to1.Λ'  …
  -/
  clear f L a R
  /-
    case intro.intro
    Γ : Type u_1
    Λ : Type u_2
    σ : Type u_3
    n : Nat
    enc : Γ → List.Vector Bool n
    dec : List.Vector Bool n → Γ
    inst✝ : Inhabited Γ
    enc0 : Eq (enc Inhabited.default) (List.Vector.replicate n Bool.false)
    encdec : ∀ (a : Γ), Eq (dec (enc a)) a
    v : σ
    ⊢ ∀ (i : Nat) (f : List.Vector Bool i → Turing.TM1.Stmt Bool Turing.TM1to1.Λ'  …
  -/
  intro i f L' R' l₁ l₂ _
  /-
    case intro.intro
    Γ : Type u_1
    Λ : Type u_2
    σ : Type u_3
    n : Nat
    enc : Γ → List.Vector Bool n
    dec : List.Vector Bool n → Γ
    inst✝ : Inhabited Γ
    enc0 : Eq (enc Inhabited.default) (List.Vector.replicate n Bool.false)
    encdec : ∀ (a : Γ), Eq (dec (enc a)) a
    v : σ
    i : Nat
    f : List.Vector Bool i → Turing.TM1.Stmt Bool Turing.TM1to1.Λ' σ
    L' R' : Turing.ListBlank Bool
    l₁ l₂ : List Bool
    h✝ : Eq l₂.length i
    ⊢ Eq (Turing.TM1.stepAux (Turing.TM1to1.readAux i f) v (Turing.Tape.mk' (Turin …
  -/
  subst i
  /-
    case intro.intro
    Γ : Type u_1
    Λ : Type u_2
    σ : Type u_3
    n : Nat
    enc : Γ → List.Vector Bool n
    dec : List.Vector Bool n → Γ
    inst✝ : Inhabited Γ
    enc0 : Eq (enc Inhabited.default) (List.Vector.replicate n Bool.false)
    encdec : ∀ (a : Γ), Eq (dec (enc a)) a
    v : σ
    L' R' : Turing.ListBlank Bool
    l₁ l₂ : List Bool
    f : List.Vector Bool l₂.length → Turing.TM1.Stmt Bool Turing.TM1to1.Λ' σ
    ⊢ Eq (Turing.TM1.stepAux (Turing.TM1to1.readAux l₂.length f) v (Turing.Tape.mk …
  -/
  induction' l₂ with a l₂ IH generalizing l₁
    /-
      case intro.intro.nil
      Γ : Type u_1
      Λ : Type u_2
      σ : Type u_3
      n : Nat
      enc : Γ → List.Vector Bool n
      dec : List.Vector Bool n → Γ
      inst✝ : Inhabited Γ
      enc0 : Eq (enc Inhabited.default) (List.Vector.replicate n Bool.false)
      encdec : ∀ (a : Γ), Eq (dec (enc a)) a
      v : σ
      L' R' : Turing.ListBlank Bool
      l₁ : List Bool
      f : List.Vector Bool List.nil.length → Turing.TM1.Stmt Bool Turing.TM1to1.Λ' σ
      ⊢ Eq (Turing.TM1.stepAux (Turing.TM1to1.readAux List.nil.length f) v (Turing.T …
    -/
  · rfl
    /-
      🎉 no goals
    -/
  trans
    stepAux (readAux l₂.length fun v ↦ f (a ::ᵥ v)) v
      (Tape.mk' ((L'.append l₁).cons a) (R'.append l₂))
    /-
      Γ : Type u_1
      Λ : Type u_2
      σ : Type u_3
      n : Nat
      enc : Γ → List.Vector Bool n
      dec : List.Vector Bool n → Γ
      inst✝ : Inhabited Γ
      enc0 : Eq (enc Inhabited.default) (List.Vector.replicate n Bool.false)
      encdec : ∀ (a : Γ), Eq (dec (enc a)) a
      v : σ
      L' R' : Turing.ListBlank Bool
      a : Bool
      l₂ : List Bool
      IH : ∀ (l₁ : List Bool) (f : List.Vector Bool l₂.length → Turing.TM1.Stmt Bool …
      l₁ : List Bool
      f : List.Vector Bool (List.cons a l₂).length → Turing.TM1.Stmt Bool Turing.TM1 …
      ⊢ Eq (Turing.TM1.stepAux (Turing.TM1to1.readAux (List.cons a l₂).length f) v ( …
    -/
  · dsimp [readAux, stepAux]
    /-
      Γ : Type u_1
      Λ : Type u_2
      σ : Type u_3
      n : Nat
      enc : Γ → List.Vector Bool n
      dec : List.Vector Bool n → Γ
      inst✝ : Inhabited Γ
      enc0 : Eq (enc Inhabited.default) (List.Vector.replicate n Bool.false)
      encdec : ∀ (a : Γ), Eq (dec (enc a)) a
      v : σ
      L' R' : Turing.ListBlank Bool
      a : Bool
      l₂ : List Bool
      IH : ∀ (l₁ : List Bool) (f : List.Vector Bool l₂.length → Turing.TM1.Stmt Bool …
      l₁ : List Bool
      f : List.Vector Bool (List.cons a l₂).length → Turing.TM1.Stmt Bool Turing.TM1 …
      ⊢ Eq (cond (Turing.ListBlank.cons a (Turing.ListBlank.append l₂ R')).head (Tur …
    -/
    simp only [ListBlank.head_cons, Tape.move_right_mk', ListBlank.tail_cons]
    /-
      Γ : Type u_1
      Λ : Type u_2
      σ : Type u_3
      n : Nat
      enc : Γ → List.Vector Bool n
      dec : List.Vector Bool n → Γ
      inst✝ : Inhabited Γ
      enc0 : Eq (enc Inhabited.default) (List.Vector.replicate n Bool.false)
      encdec : ∀ (a : Γ), Eq (dec (enc a)) a
      v : σ
      L' R' : Turing.ListBlank Bool
      a : Bool
      l₂ : List Bool
      IH : ∀ (l₁ : List Bool) (f : List.Vector Bool l₂.length → Turing.TM1.Stmt Bool …
      l₁ : List Bool
      f : List.Vector Bool (List.cons a l₂).length → Turing.TM1.Stmt Bool Turing.TM1 …
      ⊢ Eq (cond a (Turing.TM1.stepAux (Turing.TM1to1.readAux l₂.length fun v => f ( …
    -/
                /-
                  🎉 no goals
                -/
    cases a <;> rfl
                /-
                  🎉 no goals
                -/
  /-
    Γ : Type u_1
    Λ : Type u_2
    σ : Type u_3
    n : Nat
    enc : Γ → List.Vector Bool n
    dec : List.Vector Bool n → Γ
    inst✝ : Inhabited Γ
    enc0 : Eq (enc Inhabited.default) (List.Vector.replicate n Bool.false)
    encdec : ∀ (a : Γ), Eq (dec (enc a)) a
    v : σ
    L' R' : Turing.ListBlank Bool
    a : Bool
    l₂ : List Bool
    IH : ∀ (l₁ : List Bool) (f : List.Vector Bool l₂.length → Turing.TM1.Stmt Bool …
    l₁ : List Bool
    f : List.Vector Bool (List.cons a l₂).length → Turing.TM1.Stmt Bool Turing.TM1 …
    ⊢ Eq (Turing.TM1.stepAux (Turing.TM1to1.readAux l₂.length fun v => f (List.Vec …
  -/
  rw [← ListBlank.append, IH]
  /-
    Γ : Type u_1
    Λ : Type u_2
    σ : Type u_3
    n : Nat
    enc : Γ → List.Vector Bool n
    dec : List.Vector Bool n → Γ
    inst✝ : Inhabited Γ
    enc0 : Eq (enc Inhabited.default) (List.Vector.replicate n Bool.false)
    encdec : ∀ (a : Γ), Eq (dec (enc a)) a
    v : σ
    L' R' : Turing.ListBlank Bool
    a : Bool
    l₂ : List Bool
    IH : ∀ (l₁ : List Bool) (f : List.Vector Bool l₂.length → Turing.TM1.Stmt Bool …
    l₁ : List Bool
    f : List.Vector Bool (List.cons a l₂).length → Turing.TM1.Stmt Bool Turing.TM1 …
    ⊢ Eq (Turing.TM1.stepAux (f (List.Vector.cons a ⟨l₂, ⋯⟩)) v (Turing.Tape.mk' ( …
  -/
  rfl
  /-
    🎉 no goals
  -/


variable {enc0} in
theorem tr_respects :
    Respects (step M) (step (tr enc dec M)) fun c₁ c₂ ↦ trCfg enc enc0 c₁ = c₂ :=
  fun_respects.2 fun ⟨l₁, v, T⟩ ↦ by
    /-
      Γ : Type u_1
      Λ : Type u_2
      σ : Type u_3
      n : Nat
      enc : Γ → List.Vector Bool n
      dec : List.Vector Bool n → Γ
      M : Λ → Turing.TM1.Stmt Γ Λ σ
      inst✝ : Inhabited Γ
      enc0 : Eq (enc Inhabited.default) (List.Vector.replicate n Bool.false)
      encdec : ∀ (a : Γ), Eq (dec (enc a)) a
      x✝ : Turing.TM1.Cfg Γ Λ σ
      l₁ : Option Λ
      v : σ
      T : Turing.Tape Γ
      ⊢ Turing.FRespects (Turing.TM1.step (Turing.TM1to1.tr enc dec M)) (Turing.TM1t …
    -/
    obtain ⟨L, R, rfl⟩ := T.exists_mk'
    /-
      case intro.intro
      Γ : Type u_1
      Λ : Type u_2
      σ : Type u_3
      n : Nat
      enc : Γ → List.Vector Bool n
      dec : List.Vector Bool n → Γ
      M : Λ → Turing.TM1.Stmt Γ Λ σ
      inst✝ : Inhabited Γ
      enc0 : Eq (enc Inhabited.default) (List.Vector.replicate n Bool.false)
      encdec : ∀ (a : Γ), Eq (dec (enc a)) a
      x✝ : Turing.TM1.Cfg Γ Λ σ
      l₁ : Option Λ
      v : σ
      L R : Turing.ListBlank Γ
      ⊢ Turing.FRespects (Turing.TM1.step (Turing.TM1to1.tr enc dec M)) (Turing.TM1t …
    -/
    cases' l₁ with l₁
      /-
        case intro.intro.none
        Γ : Type u_1
        Λ : Type u_2
        σ : Type u_3
        n : Nat
        enc : Γ → List.Vector Bool n
        dec : List.Vector Bool n → Γ
        M : Λ → Turing.TM1.Stmt Γ Λ σ
        inst✝ : Inhabited Γ
        enc0 : Eq (enc Inhabited.default) (List.Vector.replicate n Bool.false)
        encdec : ∀ (a : Γ), Eq (dec (enc a)) a
        x✝ : Turing.TM1.Cfg Γ Λ σ
        v : σ
        L R : Turing.ListBlank Γ
        ⊢ Turing.FRespects (Turing.TM1.step (Turing.TM1to1.tr enc dec M)) (Turing.TM1t …
      -/
    · exact rfl
      /-
        🎉 no goals
      -/
    suffices ∀ q R, Reaches (step (tr enc dec M)) (stepAux (trNormal dec q) v (trTape' enc0 L R))
        (trCfg enc enc0 (stepAux q v (Tape.mk' L R))) by
      refine TransGen.head' rfl ?_
      rw [trTape_mk']
      exact this _ R
    /-
      case intro.intro.some
      Γ : Type u_1
      Λ : Type u_2
      σ : Type u_3
      n : Nat
      enc : Γ → List.Vector Bool n
      dec : List.Vector Bool n → Γ
      M : Λ → Turing.TM1.Stmt Γ Λ σ
      inst✝ : Inhabited Γ
      enc0 : Eq (enc Inhabited.default) (List.Vector.replicate n Bool.false)
      encdec : ∀ (a : Γ), Eq (dec (enc a)) a
      x✝ : Turing.TM1.Cfg Γ Λ σ
      v : σ
      L R : Turing.ListBlank Γ
      l₁ : Λ
      ⊢ ∀ (q : Turing.TM1.Stmt Γ Λ σ) (R : Turing.ListBlank Γ), Turing.Reaches (Turi …
    -/
    clear R l₁
    /-
      case intro.intro.some
      Γ : Type u_1
      Λ : Type u_2
      σ : Type u_3
      n : Nat
      enc : Γ → List.Vector Bool n
      dec : List.Vector Bool n → Γ
      M : Λ → Turing.TM1.Stmt Γ Λ σ
      inst✝ : Inhabited Γ
      enc0 : Eq (enc Inhabited.default) (List.Vector.replicate n Bool.false)
      encdec : ∀ (a : Γ), Eq (dec (enc a)) a
      x✝ : Turing.TM1.Cfg Γ Λ σ
      v : σ
      L : Turing.ListBlank Γ
      ⊢ ∀ (q : Turing.TM1.Stmt Γ Λ σ) (R : Turing.ListBlank Γ), Turing.Reaches (Turi …
    -/
    intro q R
    induction q generalizing v L R with
    | move d q IH =>
      cases d <;>
          simp only [trNormal, iterate, stepAux_move, stepAux, ListBlank.head_cons,
            Tape.move_left_mk', ListBlank.cons_head_tail, ListBlank.tail_cons,
            trTape'_move_left enc0, trTape'_move_right enc0] <;>
        apply IH
    | write f q IH =>
      simp only [trNormal, stepAux_read dec enc0 encdec, stepAux]
      refine ReflTransGen.head rfl ?_
      obtain ⟨a, R, rfl⟩ := R.exists_cons
      rw [tr, Tape.mk'_head, stepAux_write, ListBlank.head_cons, stepAux_move,
        trTape'_move_left enc0, ListBlank.head_cons, ListBlank.tail_cons, Tape.write_mk']
      apply IH
    | load a q IH =>
      simp only [trNormal, stepAux_read dec enc0 encdec]
      apply IH
    | branch p q₁ q₂ IH₁ IH₂ =>
      simp only [trNormal, stepAux_read dec enc0 encdec, stepAux, Tape.mk'_head]
      cases p R.head v <;> [apply IH₂; apply IH₁]
    | goto l =>
      simp only [trNormal, stepAux_read dec enc0 encdec, stepAux, trCfg, trTape_mk']
      apply ReflTransGen.refl
    | halt =>
      simp only [trNormal, stepAux, trCfg, stepAux_move, trTape'_move_left enc0,
        trTape'_move_right enc0, trTape_mk']
      apply ReflTransGen.refl


/-- The set of accessible `Λ'.write` machine states. -/
noncomputable def writes : Stmt₁ → Finset Λ'₁
  | Stmt.move _ q => writes q
  | Stmt.write _ q => (Finset.univ.image fun a ↦ Λ'.write a q) ∪ writes q
  | Stmt.load _ q => writes q
  | Stmt.branch _ q₁ q₂ => writes q₁ ∪ writes q₂
  | Stmt.goto _ => ∅
  | Stmt.halt => ∅


/-- The set of accessible machine states, assuming that the input machine is supported on `S`,
are the normal states embedded from `S`, plus all write states accessible from these states. -/
noncomputable def trSupp (S : Finset Λ) : Finset Λ'₁ :=
  S.biUnion fun l ↦ insert (Λ'.normal l) (writes (M l))


theorem tr_supports [Inhabited Λ] {S : Finset Λ} (ss : Supports M S) :
    Supports (tr enc dec M) (trSupp M S) :=
  ⟨Finset.mem_biUnion.2 ⟨_, ss.1, Finset.mem_insert_self _ _⟩, fun q h ↦ by
    suffices ∀ q, SupportsStmt S q → (∀ q' ∈ writes q, q' ∈ trSupp M S) →
        SupportsStmt (trSupp M S) (trNormal dec q) ∧
        ∀ q' ∈ writes q, SupportsStmt (trSupp M S) (tr enc dec M q') by
      rcases Finset.mem_biUnion.1 h with ⟨l, hl, h⟩
      have :=
        this _ (ss.2 _ hl) fun q' hq ↦ Finset.mem_biUnion.2 ⟨_, hl, Finset.mem_insert_of_mem hq⟩
      rcases Finset.mem_insert.1 h with (rfl | h)
      exacts [this.1, this.2 _ h]
    /-
      Γ : Type u_1
      Λ : Type u_2
      σ : Type u_3
      n : Nat
      enc : Γ → List.Vector Bool n
      dec : List.Vector Bool n → Γ
      M : Λ → Turing.TM1.Stmt Γ Λ σ
      inst✝¹ : Fintype Γ
      inst✝ : Inhabited Λ
      S : Finset Λ
      ss : Turing.TM1.Supports M S
      q : Turing.TM1to1.Λ'
      h : Membership.mem (Turing.TM1to1.trSupp M S) q
      ⊢ ∀ (q : Turing.TM1.Stmt Γ Λ σ), Turing.TM1.SupportsStmt S q → (∀ (q' : Turing …
    -/
    intro q hs hw
    induction q with
    | move d q IH =>
      unfold writes at hw ⊢
      replace IH := IH hs hw; refine ⟨?_, IH.2⟩
      cases d <;> simp only [trNormal, iterate, supportsStmt_move, IH]
    | write f q IH =>
      unfold writes at hw ⊢
      simp only [Finset.mem_image, Finset.mem_union, Finset.mem_univ, exists_prop, true_and]
        at hw ⊢
      replace IH := IH hs fun q hq ↦ hw q (Or.inr hq)
      refine ⟨supportsStmt_read _ fun a _ s ↦ hw _ (Or.inl ⟨_, rfl⟩), fun q' hq ↦ ?_⟩
      rcases hq with (⟨a, q₂, rfl⟩ | hq)
      · simp only [tr, supportsStmt_write, supportsStmt_move, IH.1]
      · exact IH.2 _ hq
    | load a q IH =>
      unfold writes at hw ⊢
      replace IH := IH hs hw
      exact ⟨supportsStmt_read _ fun _ ↦ IH.1, IH.2⟩
    | branch p q₁ q₂ IH₁ IH₂ =>
      unfold writes at hw ⊢
      simp only [Finset.mem_union] at hw ⊢
      replace IH₁ := IH₁ hs.1 fun q hq ↦ hw q (Or.inl hq)
      replace IH₂ := IH₂ hs.2 fun q hq ↦ hw q (Or.inr hq)
      exact ⟨supportsStmt_read _ fun _ ↦ ⟨IH₁.1, IH₂.1⟩, fun q ↦ Or.rec (IH₁.2 _) (IH₂.2 _)⟩
    | goto l =>
      simp only [writes, Finset.not_mem_empty]; refine ⟨?_, fun _ ↦ False.elim⟩
      refine supportsStmt_read _ fun a _ s ↦ ?_
      exact Finset.mem_biUnion.2 ⟨_, hs _ _, Finset.mem_insert_self _ _⟩
    | halt =>
      simp only [writes, Finset.not_mem_empty]; refine ⟨?_, fun _ ↦ False.elim⟩
      simp only [SupportsStmt, supportsStmt_move, trNormal]⟩


/-- The machine states for a TM1 emulating a TM0 machine. States of the TM0 machine are embedded
as `normal q` states, but the actual operation is split into two parts, a jump to `act s q`
followed by the action and a jump to the next `normal` state. -/
inductive Λ'
  | normal : Λ → Λ'
  | act : TM0.Stmt Γ → Λ → Λ'


local notation "Λ'₁" => @Λ' Γ Λ  -- Porting note (https://github.com/leanprover-community/mathlib4/issues/10750): added this to clean up types.


instance : Inhabited Λ'₁ :=
  ⟨Λ'.normal default⟩


local notation "Cfg₀" => TM0.Cfg Γ Λ


local notation "Stmt₁" => TM1.Stmt Γ Λ'₁ Unit


local notation "Cfg₁" => TM1.Cfg Γ Λ'₁ Unit


/-- The program. -/
def tr : Λ'₁ → Stmt₁
  | Λ'.normal q =>
    branch (fun a _ ↦ (M q a).isNone) halt <|
      goto fun a _ ↦ match M q a with
      | none => default -- unreachable
      | some (q', s) => Λ'.act s q'
  | Λ'.act (TM0.Stmt.move d) q => move d <| goto fun _ _ ↦ Λ'.normal q
  | Λ'.act (TM0.Stmt.write a) q => (write fun _ _ ↦ a) <| goto fun _ _ ↦ Λ'.normal q


/-- The configuration translation. -/
def trCfg : Cfg₀ → Cfg₁
  | ⟨q, T⟩ => ⟨cond (M q T.1).isSome (some (Λ'.normal q)) none, (), T⟩


theorem tr_respects : Respects (TM0.step M) (TM1.step (tr M)) fun a b ↦ trCfg M a = b :=
  fun_respects.2 fun ⟨q, T⟩ ↦ by
    /-
      Γ : Type u_1
      inst✝¹ : Inhabited Γ
      Λ : Type u_2
      inst✝ : Inhabited Λ
      M : Turing.TM0.Machine Γ Λ
      x✝ : Turing.TM0.Cfg Γ Λ
      q : Λ
      T : Turing.Tape Γ
      ⊢ Turing.FRespects (Turing.TM1.step (Turing.TM0to1.tr M)) (Turing.TM0to1.trCfg …
    -/
    cases' e : M q T.1 with val
      /-
        case none
        Γ : Type u_1
        inst✝¹ : Inhabited Γ
        Λ : Type u_2
        inst✝ : Inhabited Λ
        M : Turing.TM0.Machine Γ Λ
        x✝ : Turing.TM0.Cfg Γ Λ
        q : Λ
        T : Turing.Tape Γ
        e : Eq (M q T.head) Option.none
        ⊢ Turing.FRespects (Turing.TM1.step (Turing.TM0to1.tr M)) (Turing.TM0to1.trCfg …
      -/
    · simp only [TM0.step, trCfg, e]; exact Eq.refl none
                                      /-
                                        🎉 no goals
                                      -/
    /-
      case some
      Γ : Type u_1
      inst✝¹ : Inhabited Γ
      Λ : Type u_2
      inst✝ : Inhabited Λ
      M : Turing.TM0.Machine Γ Λ
      x✝ : Turing.TM0.Cfg Γ Λ
      q : Λ
      T : Turing.Tape Γ
      val : Prod Λ (Turing.TM0.Stmt Γ)
      e : Eq (M q T.head) (Option.some val)
      ⊢ Turing.FRespects (Turing.TM1.step (Turing.TM0to1.tr M)) (Turing.TM0to1.trCfg …
    -/
    cases' val with q' s
    /-
      case some.mk
      Γ : Type u_1
      inst✝¹ : Inhabited Γ
      Λ : Type u_2
      inst✝ : Inhabited Λ
      M : Turing.TM0.Machine Γ Λ
      x✝ : Turing.TM0.Cfg Γ Λ
      q : Λ
      T : Turing.Tape Γ
      q' : Λ
      s : Turing.TM0.Stmt Γ
      e : Eq (M q T.head) (Option.some { fst := q', snd := s })
      ⊢ Turing.FRespects (Turing.TM1.step (Turing.TM0to1.tr M)) (Turing.TM0to1.trCfg …
    -/
    simp only [FRespects, TM0.step, trCfg, e, Option.isSome, cond, Option.map_some']
    /-
      case some.mk
      Γ : Type u_1
      inst✝¹ : Inhabited Γ
      Λ : Type u_2
      inst✝ : Inhabited Λ
      M : Turing.TM0.Machine Γ Λ
      x✝ : Turing.TM0.Cfg Γ Λ
      q : Λ
      T : Turing.Tape Γ
      q' : Λ
      s : Turing.TM0.Stmt Γ
      e : Eq (M q T.head) (Option.some { fst := q', snd := s })
      ⊢ Turing.Reaches₁ (Turing.TM1.step (Turing.TM0to1.tr M)) { l := Option.some (T …
    -/
    revert e  -- Porting note: Added this so that `e` doesn't get into the `match`.
    have : TM1.step (tr M) ⟨some (Λ'.act s q'), (), T⟩ = some ⟨some (Λ'.normal q'), (), match s with
        | TM0.Stmt.move d => T.move d
        | TM0.Stmt.write a => T.write a⟩ := by
      cases' s with d a <;> rfl
    /-
      case some.mk
      Γ : Type u_1
      inst✝¹ : Inhabited Γ
      Λ : Type u_2
      inst✝ : Inhabited Λ
      M : Turing.TM0.Machine Γ Λ
      x✝ : Turing.TM0.Cfg Γ Λ
      q : Λ
      T : Turing.Tape Γ
      q' : Λ
      s : Turing.TM0.Stmt Γ
      this : Eq (Turing.TM1.step (Turing.TM0to1.tr M) { l := Option.some (Turing.TM0 …
      ⊢ Eq (M q T.head) (Option.some { fst := q', snd := s }) → Turing.Reaches₁ (Tur …
    -/
    intro e
    /-
      case some.mk
      Γ : Type u_1
      inst✝¹ : Inhabited Γ
      Λ : Type u_2
      inst✝ : Inhabited Λ
      M : Turing.TM0.Machine Γ Λ
      x✝ : Turing.TM0.Cfg Γ Λ
      q : Λ
      T : Turing.Tape Γ
      q' : Λ
      s : Turing.TM0.Stmt Γ
      this : Eq (Turing.TM1.step (Turing.TM0to1.tr M) { l := Option.some (Turing.TM0 …
      e : Eq (M q T.head) (Option.some { fst := q', snd := s })
      ⊢ Turing.Reaches₁ (Turing.TM1.step (Turing.TM0to1.tr M)) { l := Option.some (T …
    -/
    refine TransGen.head ?_ (TransGen.head' this ?_)
      /-
        case some.mk.refine_1
        Γ : Type u_1
        inst✝¹ : Inhabited Γ
        Λ : Type u_2
        inst✝ : Inhabited Λ
        M : Turing.TM0.Machine Γ Λ
        x✝ : Turing.TM0.Cfg Γ Λ
        q : Λ
        T : Turing.Tape Γ
        q' : Λ
        s : Turing.TM0.Stmt Γ
        this : Eq (Turing.TM1.step (Turing.TM0to1.tr M) { l := Option.some (Turing.TM0 …
        e : Eq (M q T.head) (Option.some { fst := q', snd := s })
        ⊢ Membership.mem (Turing.TM1.step (Turing.TM0to1.tr M) { l := Option.some (Tur …
      -/
    · simp only [TM1.step, TM1.stepAux]
      /-
        case some.mk.refine_1
        Γ : Type u_1
        inst✝¹ : Inhabited Γ
        Λ : Type u_2
        inst✝ : Inhabited Λ
        M : Turing.TM0.Machine Γ Λ
        x✝ : Turing.TM0.Cfg Γ Λ
        q : Λ
        T : Turing.Tape Γ
        q' : Λ
        s : Turing.TM0.Stmt Γ
        this : Eq (Turing.TM1.step (Turing.TM0to1.tr M) { l := Option.some (Turing.TM0 …
        e : Eq (M q T.head) (Option.some { fst := q', snd := s })
        ⊢ Membership.mem (Option.some (cond (M q T.head).isNone { l := Option.none, va …
      -/
      rw [e]
      /-
        case some.mk.refine_1
        Γ : Type u_1
        inst✝¹ : Inhabited Γ
        Λ : Type u_2
        inst✝ : Inhabited Λ
        M : Turing.TM0.Machine Γ Λ
        x✝ : Turing.TM0.Cfg Γ Λ
        q : Λ
        T : Turing.Tape Γ
        q' : Λ
        s : Turing.TM0.Stmt Γ
        this : Eq (Turing.TM1.step (Turing.TM0to1.tr M) { l := Option.some (Turing.TM0 …
        e : Eq (M q T.head) (Option.some { fst := q', snd := s })
        ⊢ Membership.mem (Option.some (cond (Option.some { fst := q', snd := s }).isNo …
      -/
      rfl
      /-
        🎉 no goals
      -/
    /-
      case some.mk.refine_2
      Γ : Type u_1
      inst✝¹ : Inhabited Γ
      Λ : Type u_2
      inst✝ : Inhabited Λ
      M : Turing.TM0.Machine Γ Λ
      x✝ : Turing.TM0.Cfg Γ Λ
      q : Λ
      T : Turing.Tape Γ
      q' : Λ
      s : Turing.TM0.Stmt Γ
      this : Eq (Turing.TM1.step (Turing.TM0to1.tr M) { l := Option.some (Turing.TM0 …
      e : Eq (M q T.head) (Option.some { fst := q', snd := s })
      ⊢ Relation.ReflTransGen (fun a b => Membership.mem (Turing.TM1.step (Turing.TM …
    -/
    cases e' : M q' _
      /-
        case some.mk.refine_2.none
        Γ : Type u_1
        inst✝¹ : Inhabited Γ
        Λ : Type u_2
        inst✝ : Inhabited Λ
        M : Turing.TM0.Machine Γ Λ
        x✝ : Turing.TM0.Cfg Γ Λ
        q : Λ
        T : Turing.Tape Γ
        q' : Λ
        s : Turing.TM0.Stmt Γ
        this : Eq (Turing.TM1.step (Turing.TM0to1.tr M) { l := Option.some (Turing.TM0 …
        e : Eq (M q T.head) (Option.some { fst := q', snd := s })
        e' : Eq (M q' (Turing.TM0.step.match_1 (fun a => Turing.Tape Γ) s (fun d => Tu …
        ⊢ Relation.ReflTransGen (fun a b => Membership.mem (Turing.TM1.step (Turing.TM …
      -/
    · apply ReflTransGen.single
      /-
        case some.mk.refine_2.none.hab
        Γ : Type u_1
        inst✝¹ : Inhabited Γ
        Λ : Type u_2
        inst✝ : Inhabited Λ
        M : Turing.TM0.Machine Γ Λ
        x✝ : Turing.TM0.Cfg Γ Λ
        q : Λ
        T : Turing.Tape Γ
        q' : Λ
        s : Turing.TM0.Stmt Γ
        this : Eq (Turing.TM1.step (Turing.TM0to1.tr M) { l := Option.some (Turing.TM0 …
        e : Eq (M q T.head) (Option.some { fst := q', snd := s })
        e' : Eq (M q' (Turing.TM0.step.match_1 (fun a => Turing.Tape Γ) s (fun d => Tu …
        ⊢ Membership.mem (Turing.TM1.step (Turing.TM0to1.tr M) { l := Option.some (Tur …
      -/
      simp only [TM1.step, TM1.stepAux]
      /-
        case some.mk.refine_2.none.hab
        Γ : Type u_1
        inst✝¹ : Inhabited Γ
        Λ : Type u_2
        inst✝ : Inhabited Λ
        M : Turing.TM0.Machine Γ Λ
        x✝ : Turing.TM0.Cfg Γ Λ
        q : Λ
        T : Turing.Tape Γ
        q' : Λ
        s : Turing.TM0.Stmt Γ
        this : Eq (Turing.TM1.step (Turing.TM0to1.tr M) { l := Option.some (Turing.TM0 …
        e : Eq (M q T.head) (Option.some { fst := q', snd := s })
        e' : Eq (M q' (Turing.TM0.step.match_1 (fun a => Turing.Tape Γ) s (fun d => Tu …
        ⊢ Membership.mem (Option.some (cond (M q' (Turing.TM0.step.match_1 (fun s => T …
      -/
      rw [e']
      /-
        case some.mk.refine_2.none.hab
        Γ : Type u_1
        inst✝¹ : Inhabited Γ
        Λ : Type u_2
        inst✝ : Inhabited Λ
        M : Turing.TM0.Machine Γ Λ
        x✝ : Turing.TM0.Cfg Γ Λ
        q : Λ
        T : Turing.Tape Γ
        q' : Λ
        s : Turing.TM0.Stmt Γ
        this : Eq (Turing.TM1.step (Turing.TM0to1.tr M) { l := Option.some (Turing.TM0 …
        e : Eq (M q T.head) (Option.some { fst := q', snd := s })
        e' : Eq (M q' (Turing.TM0.step.match_1 (fun a => Turing.Tape Γ) s (fun d => Tu …
        ⊢ Membership.mem (Option.some (cond Option.none.isNone { l := Option.none, var …
      -/
      rfl
      /-
        🎉 no goals
      -/
      /-
        case some.mk.refine_2.some
        Γ : Type u_1
        inst✝¹ : Inhabited Γ
        Λ : Type u_2
        inst✝ : Inhabited Λ
        M : Turing.TM0.Machine Γ Λ
        x✝ : Turing.TM0.Cfg Γ Λ
        q : Λ
        T : Turing.Tape Γ
        q' : Λ
        s : Turing.TM0.Stmt Γ
        this : Eq (Turing.TM1.step (Turing.TM0to1.tr M) { l := Option.some (Turing.TM0 …
        e : Eq (M q T.head) (Option.some { fst := q', snd := s })
        val✝ : Prod Λ (Turing.TM0.Stmt Γ)
        e' : Eq (M q' (Turing.TM0.step.match_1 (fun a => Turing.Tape Γ) s (fun d => Tu …
        ⊢ Relation.ReflTransGen (fun a b => Membership.mem (Turing.TM1.step (Turing.TM …
      -/
    · rfl
      /-
        🎉 no goals
      -/


/-- The TM2 model removes the tape entirely from the TM1 model,
  replacing it with an arbitrary (finite) collection of stacks.
  The operation `push` puts an element on one of the stacks,
  and `pop` removes an element from a stack (and modifying the
  internal state based on the result). `peek` modifies the
  internal state but does not remove an element. -/
inductive Stmt
  | push : ∀ k, (σ → Γ k) → Stmt → Stmt
  | peek : ∀ k, (σ → Option (Γ k) → σ) → Stmt → Stmt
  | pop : ∀ k, (σ → Option (Γ k) → σ) → Stmt → Stmt
  | load : (σ → σ) → Stmt → Stmt
  | branch : (σ → Bool) → Stmt → Stmt → Stmt
  | goto : (σ → Λ) → Stmt
  | halt : Stmt


local notation "Stmt₂" => Stmt Γ Λ σ  -- Porting note (https://github.com/leanprover-community/mathlib4/issues/10750): added this to clean up types.


instance Stmt.inhabited : Inhabited Stmt₂ :=
  ⟨halt⟩


/-- A configuration in the TM2 model is a label (or `none` for the halt state), the state of
local variables, and the stacks. (Note that the stacks are not `ListBlank`s, they have a definite
size.) -/
structure Cfg where
  /-- The current label to run (or `none` for the halting state) -/
  l : Option Λ
  /-- The internal state -/
  var : σ
  /-- The (finite) collection of internal stacks -/
  stk : ∀ k, List (Γ k)


local notation "Cfg₂" => Cfg Γ Λ σ  -- Porting note (https://github.com/leanprover-community/mathlib4/issues/10750): added this to clean up types.


instance Cfg.inhabited [Inhabited σ] : Inhabited Cfg₂ :=
  ⟨⟨default, default, default⟩⟩


/-- The step function for the TM2 model. -/
@[simp]
def stepAux : Stmt₂ → σ → (∀ k, List (Γ k)) → Cfg₂
  | push k f q, v, S => stepAux q v (update S k (f v :: S k))
  | peek k f q, v, S => stepAux q (f v (S k).head?) S
  | pop k f q, v, S => stepAux q (f v (S k).head?) (update S k (S k).tail)
  | load a q, v, S => stepAux q (a v) S
  | branch f q₁ q₂, v, S => cond (f v) (stepAux q₁ v S) (stepAux q₂ v S)
  | goto f, v, S => ⟨some (f v), v, S⟩
  | halt, v, S => ⟨none, v, S⟩


/-- The step function for the TM2 model. -/
@[simp]
def step (M : Λ → Stmt₂) : Cfg₂ → Option Cfg₂
  | ⟨none, _, _⟩ => none
  | ⟨some l, v, S⟩ => some (stepAux (M l) v S)


/-- The (reflexive) reachability relation for the TM2 model. -/
def Reaches (M : Λ → Stmt₂) : Cfg₂ → Cfg₂ → Prop :=
  ReflTransGen fun a b ↦ b ∈ step M a


/-- Given a set `S` of states, `SupportsStmt S q` means that `q` only jumps to states in `S`. -/
def SupportsStmt (S : Finset Λ) : Stmt₂ → Prop
  | push _ _ q => SupportsStmt S q
  | peek _ _ q => SupportsStmt S q
  | pop _ _ q => SupportsStmt S q
  | load _ q => SupportsStmt S q
  | branch _ q₁ q₂ => SupportsStmt S q₁ ∧ SupportsStmt S q₂
  | goto l => ∀ v, l v ∈ S
  | halt => True


/-- The set of subtree statements in a statement. -/
noncomputable def stmts₁ : Stmt₂ → Finset Stmt₂
  | Q@(push _ _ q) => insert Q (stmts₁ q)
  | Q@(peek _ _ q) => insert Q (stmts₁ q)
  | Q@(pop _ _ q) => insert Q (stmts₁ q)
  | Q@(load _ q) => insert Q (stmts₁ q)
  | Q@(branch _ q₁ q₂) => insert Q (stmts₁ q₁ ∪ stmts₁ q₂)
  | Q@(goto _) => {Q}
  | Q@halt => {Q}


theorem stmts₁_self {q : Stmt₂} : q ∈ stmts₁ q := by
  /-
    K : Type u_1
    Γ : K → Type u_2
    Λ : Type u_3
    σ : Type u_4
    q : Turing.TM2.Stmt Γ Λ σ
    ⊢ Membership.mem (Turing.TM2.stmts₁ q) q
  -/
              /-
                🎉 no goals
              -/
              /-
                🎉 no goals
              -/
              /-
                🎉 no goals
              -/
              /-
                🎉 no goals
              -/
              /-
                🎉 no goals
              -/
              /-
                🎉 no goals
              -/
  cases q <;> simp only [Finset.mem_insert_self, Finset.mem_singleton_self, stmts₁]
              /-
                🎉 no goals
              -/


theorem stmts₁_trans {q₁ q₂ : Stmt₂} : q₁ ∈ stmts₁ q₂ → stmts₁ q₁ ⊆ stmts₁ q₂ := by
  /-
    K : Type u_1
    Γ : K → Type u_2
    Λ : Type u_3
    σ : Type u_4
    q₁ q₂ : Turing.TM2.Stmt Γ Λ σ
    ⊢ Membership.mem (Turing.TM2.stmts₁ q₂) q₁ → HasSubset.Subset (Turing.TM2.stmt …
  -/
  intro h₁₂ q₀ h₀₁
  induction q₂ with (
    simp only [stmts₁] at h₁₂ ⊢
    simp only [Finset.mem_insert, Finset.mem_singleton, Finset.mem_union] at h₁₂)
  | branch f q₁ q₂ IH₁ IH₂ =>
    rcases h₁₂ with (rfl | h₁₂ | h₁₂)
    · unfold stmts₁ at h₀₁
      exact h₀₁
    · exact Finset.mem_insert_of_mem (Finset.mem_union_left _ (IH₁ h₁₂))
    · exact Finset.mem_insert_of_mem (Finset.mem_union_right _ (IH₂ h₁₂))
  | goto l => subst h₁₂; exact h₀₁
  | halt => subst h₁₂; exact h₀₁
  | load  _ q IH | _ _ _ q IH =>
    rcases h₁₂ with (rfl | h₁₂)
    · unfold stmts₁ at h₀₁
      exact h₀₁
    · exact Finset.mem_insert_of_mem (IH h₁₂)


theorem stmts₁_supportsStmt_mono {S : Finset Λ} {q₁ q₂ : Stmt₂} (h : q₁ ∈ stmts₁ q₂)
    (hs : SupportsStmt S q₂) : SupportsStmt S q₁ := by
  induction q₂ with
    simp only [stmts₁, SupportsStmt, Finset.mem_insert, Finset.mem_union, Finset.mem_singleton]
      at h hs
  | branch f q₁ q₂ IH₁ IH₂ => rcases h with (rfl | h | h); exacts [hs, IH₁ h hs.1, IH₂ h hs.2]
  | goto l => subst h; exact hs
  | halt => subst h; trivial
  | load _ _ IH | _ _ _ _ IH => rcases h with (rfl | h) <;> [exact hs; exact IH h hs]


/-- The set of statements accessible from initial set `S` of labels. -/
noncomputable def stmts (M : Λ → Stmt₂) (S : Finset Λ) : Finset (Option Stmt₂) :=
  Finset.insertNone (S.biUnion fun q ↦ stmts₁ (M q))


theorem stmts_trans {M : Λ → Stmt₂} {S : Finset Λ} {q₁ q₂ : Stmt₂} (h₁ : q₁ ∈ stmts₁ q₂) :
    some q₂ ∈ stmts M S → some q₁ ∈ stmts M S := by
  simp only [stmts, Finset.mem_insertNone, Finset.mem_biUnion, Option.mem_def, Option.some.injEq,
    forall_eq', exists_imp, and_imp]
  /-
    K : Type u_1
    Γ : K → Type u_2
    Λ : Type u_3
    σ : Type u_4
    M : Λ → Turing.TM2.Stmt Γ Λ σ
    S : Finset Λ
    q₁ q₂ : Turing.TM2.Stmt Γ Λ σ
    h₁ : Membership.mem (Turing.TM2.stmts₁ q₂) q₁
    ⊢ ∀ (x : Λ), Membership.mem S x → Membership.mem (Turing.TM2.stmts₁ (M x)) q₂  …
  -/
  exact fun l ls h₂ ↦ ⟨_, ls, stmts₁_trans h₂ h₁⟩
  /-
    🎉 no goals
  -/


/-- Given a TM2 machine `M` and a set `S` of states, `Supports M S` means that all states in
`S` jump only to other states in `S`. -/
def Supports (M : Λ → Stmt₂) (S : Finset Λ) :=
  default ∈ S ∧ ∀ q ∈ S, SupportsStmt S (M q)


theorem stmts_supportsStmt {M : Λ → Stmt₂} {S : Finset Λ} {q : Stmt₂} (ss : Supports M S) :
    some q ∈ stmts M S → SupportsStmt S q := by
  simp only [stmts, Finset.mem_insertNone, Finset.mem_biUnion, Option.mem_def, Option.some.injEq,
    forall_eq', exists_imp, and_imp]
  /-
    K : Type u_1
    Γ : K → Type u_2
    Λ : Type u_3
    σ : Type u_4
    inst✝ : Inhabited Λ
    M : Λ → Turing.TM2.Stmt Γ Λ σ
    S : Finset Λ
    q : Turing.TM2.Stmt Γ Λ σ
    ss : Turing.TM2.Supports M S
    ⊢ ∀ (x : Λ), Membership.mem S x → Membership.mem (Turing.TM2.stmts₁ (M x)) q → …
  -/
  exact fun l ls h ↦ stmts₁_supportsStmt_mono h (ss.2 _ ls)
  /-
    🎉 no goals
  -/


theorem step_supports (M : Λ → Stmt₂) {S : Finset Λ} (ss : Supports M S) :
    ∀ {c c' : Cfg₂}, c' ∈ step M c → c.l ∈ Finset.insertNone S → c'.l ∈ Finset.insertNone S
  | ⟨some l₁, v, T⟩, c', h₁, h₂ => by
    /-
      K : Type u_1
      Γ : K → Type u_2
      Λ : Type u_3
      σ : Type u_4
      inst✝¹ : Inhabited Λ
      inst✝ : DecidableEq K
      M : Λ → Turing.TM2.Stmt Γ Λ σ
      S : Finset Λ
      ss : Turing.TM2.Supports M S
      l₁ : Λ
      v : σ
      T : (k : K) → List (Γ k)
      c' : Turing.TM2.Cfg Γ Λ σ
      h₁ : Membership.mem (Turing.TM2.step M { l := Option.some l₁, var := v, stk := …
      h₂ : Membership.mem (Finset.insertNone S) { l := Option.some l₁, var := v, stk …
      ⊢ Membership.mem (Finset.insertNone S) c'.l
    -/
    replace h₂ := ss.2 _ (Finset.some_mem_insertNone.1 h₂)
    /-
      K : Type u_1
      Γ : K → Type u_2
      Λ : Type u_3
      σ : Type u_4
      inst✝¹ : Inhabited Λ
      inst✝ : DecidableEq K
      M : Λ → Turing.TM2.Stmt Γ Λ σ
      S : Finset Λ
      ss : Turing.TM2.Supports M S
      l₁ : Λ
      v : σ
      T : (k : K) → List (Γ k)
      c' : Turing.TM2.Cfg Γ Λ σ
      h₁ : Membership.mem (Turing.TM2.step M { l := Option.some l₁, var := v, stk := …
      h₂ : Turing.TM2.SupportsStmt S (M l₁)
      ⊢ Membership.mem (Finset.insertNone S) c'.l
    -/
    simp only [step, Option.mem_def, Option.some.injEq] at h₁; subst c'
    /-
      K : Type u_1
      Γ : K → Type u_2
      Λ : Type u_3
      σ : Type u_4
      inst✝¹ : Inhabited Λ
      inst✝ : DecidableEq K
      M : Λ → Turing.TM2.Stmt Γ Λ σ
      S : Finset Λ
      ss : Turing.TM2.Supports M S
      l₁ : Λ
      v : σ
      T : (k : K) → List (Γ k)
      h₂ : Turing.TM2.SupportsStmt S (M l₁)
      ⊢ Membership.mem (Finset.insertNone S) (Turing.TM2.stepAux (M l₁) v T).l
    -/
    revert h₂; induction M l₁ generalizing v T with intro hs
    | branch p q₁' q₂' IH₁ IH₂ =>
      unfold stepAux; cases p v
      · exact IH₂ _ _ hs.2
      · exact IH₁ _ _ hs.1
    | goto => exact Finset.some_mem_insertNone.2 (hs _)
    | halt => apply Multiset.mem_cons_self
    | load _ _ IH | _ _ _ _ IH => exact IH _ _ hs


/-- The initial state of the TM2 model. The input is provided on a designated stack. -/
def init (k : K) (L : List (Γ k)) : Cfg₂ :=
  ⟨some default, default, update (fun _ ↦ []) k L⟩


/-- Evaluates a TM2 program to completion, with the output on the same stack as the input. -/
def eval (M : Λ → Stmt₂) (k : K) (L : List (Γ k)) : Part (List (Γ k)) :=
  (Turing.eval (step M) (init k L)).map fun c ↦ c.stk k


theorem stk_nth_val {K : Type*} {Γ : K → Type*} {L : ListBlank (∀ k, Option (Γ k))} {k S} (n)
    (hL : ListBlank.map (proj k) L = ListBlank.mk (List.map some S).reverse) :
    L.nth n k = S.reverse[n]? := by
  rw [← proj_map_nth, hL, ← List.map_reverse, ListBlank.nth_mk,
    List.getI_eq_iget_getElem?, List.getElem?_map]
  /-
    K : Type u_1
    Γ : K → Type u_2
    L : Turing.ListBlank ((k : K) → Option (Γ k))
    k : K
    S : List (Γ k)
    n : Nat
    hL : Eq (Turing.ListBlank.map (Turing.proj k) L) (Turing.ListBlank.mk (List.ma …
    ⊢ Eq (Option.map Option.some (GetElem?.getElem? S.reverse n)).iget (GetElem?.g …
  -/
                          /-
                            🎉 no goals
                          -/
  cases S.reverse[n]? <;> rfl
                          /-
                            🎉 no goals
                          -/


local notation "Stmt₂" => TM2.Stmt Γ Λ σ


local notation "Cfg₂" => TM2.Cfg Γ Λ σ

-- Porting note: `DecidableEq K` is not necessary.

/-- The alphabet of the TM2 simulator on TM1 is a marker for the stack bottom,
plus a vector of stack elements for each stack, or none if the stack does not extend this far. -/
def Γ' :=
  Bool × ∀ k, Option (Γ k)


local notation "Γ'₂₁" => @Γ' K Γ  -- Porting note (https://github.com/leanprover-community/mathlib4/issues/10750): added this to clean up types.


instance Γ'.inhabited : Inhabited Γ'₂₁ :=
  ⟨⟨false, fun _ ↦ none⟩⟩


instance Γ'.fintype [DecidableEq K] [Fintype K] [∀ k, Fintype (Γ k)] : Fintype Γ'₂₁ :=
  instFintypeProd _ _


/-- The bottom marker is fixed throughout the calculation, so we use the `addBottom` function
to express the program state in terms of a tape with only the stacks themselves. -/
def addBottom (L : ListBlank (∀ k, Option (Γ k))) : ListBlank Γ'₂₁ :=
  ListBlank.cons (true, L.head) (L.tail.map ⟨Prod.mk false, rfl⟩)


theorem addBottom_map (L : ListBlank (∀ k, Option (Γ k))) :
                                    /-
                                      K : Type u_1
                                      Γ : K → Type u_2
                                      Λ : Type u_3
                                      σ : Type u_4
                                      L : Turing.ListBlank ((k : K) → Option (Γ k))
                                      ⊢ Eq Inhabited.default.2 Inhabited.default
                                    -/
    (addBottom L).map ⟨Prod.snd, by rfl⟩ = L := by
                                    /-
                                      🎉 no goals
                                    -/
  /-
    K : Type u_1
    Γ : K → Type u_2
    L : Turing.ListBlank ((k : K) → Option (Γ k))
    ⊢ Eq (Turing.ListBlank.map { f := Prod.snd, map_pt' := ⋯ } (Turing.TM2to1.addB …
  -/
  simp only [addBottom, ListBlank.map_cons]
  /-
    K : Type u_1
    Γ : K → Type u_2
    L : Turing.ListBlank ((k : K) → Option (Γ k))
    ⊢ Eq (Turing.ListBlank.cons L.head (Turing.ListBlank.map { f := Prod.snd, map_ …
  -/
  convert ListBlank.cons_head_tail L
  /-
    case h.e'_2.h.e'_4
    K : Type u_1
    Γ : K → Type u_2
    L : Turing.ListBlank ((k : K) → Option (Γ k))
    ⊢ Eq (Turing.ListBlank.map { f := Prod.snd, map_pt' := ⋯ } (Turing.ListBlank.m …
  -/
  generalize ListBlank.tail L = L'
  /-
    case h.e'_2.h.e'_4
    K : Type u_1
    Γ : K → Type u_2
    L L' : Turing.ListBlank ((k : K) → Option (Γ k))
    ⊢ Eq (Turing.ListBlank.map { f := Prod.snd, map_pt' := ⋯ } (Turing.ListBlank.m …
  -/
  refine L'.induction_on fun l ↦ ?_; simp
                                     /-
                                       🎉 no goals
                                     -/


theorem addBottom_modifyNth (f : (∀ k, Option (Γ k)) → ∀ k, Option (Γ k))
    (L : ListBlank (∀ k, Option (Γ k))) (n : ℕ) :
    (addBottom L).modifyNth (fun a ↦ (a.1, f a.2)) n = addBottom (L.modifyNth f n) := by
  /-
    K : Type u_1
    Γ : K → Type u_2
    f : ((k : K) → Option (Γ k)) → (k : K) → Option (Γ k)
    L : Turing.ListBlank ((k : K) → Option (Γ k))
    n : Nat
    ⊢ Eq (Turing.ListBlank.modifyNth (fun a => { fst := a.1, snd := f a.2 }) n (Tu …
  -/
  cases n <;>
    /-
      case zero
      K : Type u_1
      Γ : K → Type u_2
      f : ((k : K) → Option (Γ k)) → (k : K) → Option (Γ k)
      L : Turing.ListBlank ((k : K) → Option (Γ k))
      ⊢ Eq (Turing.ListBlank.modifyNth (fun a => { fst := a.1, snd := f a.2 }) 0 (Tu …
    -/
    /-
      🎉 no goals
    -/
    simp only [addBottom, ListBlank.head_cons, ListBlank.modifyNth, ListBlank.tail_cons]
  /-
    case succ
    K : Type u_1
    Γ : K → Type u_2
    f : ((k : K) → Option (Γ k)) → (k : K) → Option (Γ k)
    L : Turing.ListBlank ((k : K) → Option (Γ k))
    n✝ : Nat
    ⊢ Eq (Turing.ListBlank.cons { fst := Bool.true, snd := L.head } (Turing.ListBl …
  -/
  congr; symm; apply ListBlank.map_modifyNth; intro; rfl
                                                     /-
                                                       🎉 no goals
                                                     -/


theorem addBottom_nth_snd (L : ListBlank (∀ k, Option (Γ k))) (n : ℕ) :
    ((addBottom L).nth n).2 = L.nth n := by
  /-
    K : Type u_1
    Γ : K → Type u_2
    L : Turing.ListBlank ((k : K) → Option (Γ k))
    n : Nat
    ⊢ Eq ((Turing.TM2to1.addBottom L).nth n).2 (L.nth n)
  -/
  conv => rhs; rw [← addBottom_map L, ListBlank.nth_map]
  /-
    🎉 no goals
  -/


theorem addBottom_nth_succ_fst (L : ListBlank (∀ k, Option (Γ k))) (n : ℕ) :
    ((addBottom L).nth (n + 1)).1 = false := by
  /-
    K : Type u_1
    Γ : K → Type u_2
    L : Turing.ListBlank ((k : K) → Option (Γ k))
    n : Nat
    ⊢ Eq ((Turing.TM2to1.addBottom L).nth (HAdd.hAdd n 1)).1 Bool.false
  -/
  rw [ListBlank.nth_succ, addBottom, ListBlank.tail_cons, ListBlank.nth_map]
  /-
    🎉 no goals
  -/


theorem addBottom_head_fst (L : ListBlank (∀ k, Option (Γ k))) : (addBottom L).head.1 = true := by
  /-
    K : Type u_1
    Γ : K → Type u_2
    L : Turing.ListBlank ((k : K) → Option (Γ k))
    ⊢ Eq (Turing.TM2to1.addBottom L).head.1 Bool.true
  -/
  rw [addBottom, ListBlank.head_cons]
  /-
    🎉 no goals
  -/


/-- A stack action is a command that interacts with the top of a stack. Our default position
is at the bottom of all the stacks, so we have to hold on to this action while going to the end
to modify the stack. -/
inductive StAct (k : K)
  | push : (σ → Γ k) → StAct k
  | peek : (σ → Option (Γ k) → σ) → StAct k
  | pop : (σ → Option (Γ k) → σ) → StAct k


local notation "StAct₂" => @StAct K Γ σ  -- Porting note (https://github.com/leanprover-community/mathlib4/issues/10750): added this to clean up types.


instance StAct.inhabited {k : K} : Inhabited (StAct₂ k) :=
  ⟨StAct.peek fun s _ ↦ s⟩


/-- The TM2 statement corresponding to a stack action. -/
def stRun {k : K} : StAct₂ k → Stmt₂ → Stmt₂
  | push f => TM2.Stmt.push k f
  | peek f => TM2.Stmt.peek k f
  | pop f => TM2.Stmt.pop k f


/-- The effect of a stack action on the local variables, given the value of the stack. -/
def stVar {k : K} (v : σ) (l : List (Γ k)) : StAct₂ k → σ
  | push _ => v
  | peek f => f v l.head?
  | pop f => f v l.head?


/-- The effect of a stack action on the stack. -/
def stWrite {k : K} (v : σ) (l : List (Γ k)) : StAct₂ k → List (Γ k)
  | push f => f v :: l
  | peek _ => l
  | pop _ => l.tail


/-- We have partitioned the TM2 statements into "stack actions", which require going to the end
of the stack, and all other actions, which do not. This is a modified recursor which lumps the
stack actions into one. -/
@[elab_as_elim]
def stmtStRec.{l} {C : Stmt₂ → Sort l} (H₁ : ∀ (k) (s : StAct₂ k) (q) (_ : C q), C (stRun s q))
    (H₂ : ∀ (a q) (_ : C q), C (TM2.Stmt.load a q))
    (H₃ : ∀ (p q₁ q₂) (_ : C q₁) (_ : C q₂), C (TM2.Stmt.branch p q₁ q₂))
    (H₄ : ∀ l, C (TM2.Stmt.goto l)) (H₅ : C TM2.Stmt.halt) : ∀ n, C n
  | TM2.Stmt.push _ f q => H₁ _ (push f) _ (stmtStRec H₁ H₂ H₃ H₄ H₅ q)
  | TM2.Stmt.peek _ f q => H₁ _ (peek f) _ (stmtStRec H₁ H₂ H₃ H₄ H₅ q)
  | TM2.Stmt.pop _ f q => H₁ _ (pop f) _ (stmtStRec H₁ H₂ H₃ H₄ H₅ q)
  | TM2.Stmt.load _ q => H₂ _ _ (stmtStRec H₁ H₂ H₃ H₄ H₅ q)
  | TM2.Stmt.branch _ q₁ q₂ => H₃ _ _ _ (stmtStRec H₁ H₂ H₃ H₄ H₅ q₁) (stmtStRec H₁ H₂ H₃ H₄ H₅ q₂)
  | TM2.Stmt.goto _ => H₄ _
  | TM2.Stmt.halt => H₅


theorem supports_run (S : Finset Λ) {k : K} (s : StAct₂ k) (q : Stmt₂) :
    TM2.SupportsStmt S (stRun s q) ↔ TM2.SupportsStmt S q := by
  /-
    K : Type u_1
    Γ : K → Type u_2
    Λ : Type u_3
    σ : Type u_4
    S : Finset Λ
    k : K
    s : Turing.TM2to1.StAct k
    q : Turing.TM2.Stmt Γ Λ σ
    ⊢ Iff (Turing.TM2.SupportsStmt S (Turing.TM2to1.stRun s q)) (Turing.TM2.Suppor …
  -/
              /-
                🎉 no goals
              -/
              /-
                🎉 no goals
              -/
  cases s <;> rfl
              /-
                🎉 no goals
              -/


/-- The machine states of the TM2 emulator. We can either be in a normal state when waiting for the
next TM2 action, or we can be in the "go" and "return" states to go to the top of the stack and
return to the bottom, respectively. -/
inductive Λ'
  | normal : Λ → Λ'
  | go (k : K) : StAct₂ k → Stmt₂ → Λ'
  | ret : Stmt₂ → Λ'


local notation "Λ'₂₁" => @Λ' K Γ Λ σ  -- Porting note (https://github.com/leanprover-community/mathlib4/issues/10750): added this to clean up types.


instance Λ'.inhabited [Inhabited Λ] : Inhabited Λ'₂₁ :=
  ⟨normal default⟩


local notation "Stmt₂₁" => TM1.Stmt Γ'₂₁ Λ'₂₁ σ

local notation "Cfg₂₁" => TM1.Cfg Γ'₂₁ Λ'₂₁ σ


/-- The program corresponding to state transitions at the end of a stack. Here we start out just
after the top of the stack, and should end just after the new top of the stack. -/
def trStAct {k : K} (q : Stmt₂₁) : StAct₂ k → Stmt₂₁
  | StAct.push f => (write fun a s ↦ (a.1, update a.2 k <| some <| f s)) <| move Dir.right q
  | StAct.peek f => move Dir.left <| (load fun a s ↦ f s (a.2 k)) <| move Dir.right q
  | StAct.pop f =>
    branch (fun a _ ↦ a.1) (load (fun _ s ↦ f s none) q)
      (move Dir.left <|
        (load fun a s ↦ f s (a.2 k)) <| write (fun a _ ↦ (a.1, update a.2 k none)) q)


/-- The initial state for the TM2 emulator, given an initial TM2 state. All stacks start out empty
except for the input stack, and the stack bottom mark is set at the head. -/
def trInit (k : K) (L : List (Γ k)) : List Γ'₂₁ :=
  let L' : List Γ'₂₁ := L.reverse.map fun a ↦ (false, update (fun _ ↦ none) k (some a))
  (true, L'.headI.2) :: L'.tail


theorem step_run {k : K} (q : Stmt₂) (v : σ) (S : ∀ k, List (Γ k)) : ∀ s : StAct₂ k,
    TM2.stepAux (stRun s q) v S = TM2.stepAux q (stVar v (S k) s) (update S k (stWrite v (S k) s))
  | StAct.push _ => rfl
                       /-
                         K : Type u_1
                         Γ : K → Type u_2
                         Λ : Type u_3
                         σ : Type u_4
                         inst✝ : DecidableEq K
                         k : K
                         q : Turing.TM2.Stmt Γ Λ σ
                         v : σ
                         S : (k : K) → List (Γ k)
                         f : σ → Option (Γ k) → σ
                         ⊢ Eq (Turing.TM2.stepAux (Turing.TM2to1.stRun (Turing.TM2to1.StAct.peek f) q)  …
                       -/
  | StAct.peek f => by unfold stWrite; rw [Function.update_eq_self]; rfl
                                                                     /-
                                                                       🎉 no goals
                                                                     -/
  | StAct.pop _ => rfl


/-- The translation of TM2 statements to TM1 statements. regular actions have direct equivalents,
but stack actions are deferred by going to the corresponding `go` state, so that we can find the
appropriate stack top. -/
def trNormal : Stmt₂ → Stmt₂₁
  | TM2.Stmt.push k f q => goto fun _ _ ↦ go k (StAct.push f) q
  | TM2.Stmt.peek k f q => goto fun _ _ ↦ go k (StAct.peek f) q
  | TM2.Stmt.pop k f q => goto fun _ _ ↦ go k (StAct.pop f) q
  | TM2.Stmt.load a q => load (fun _ ↦ a) (trNormal q)
  | TM2.Stmt.branch f q₁ q₂ => branch (fun _ ↦ f) (trNormal q₁) (trNormal q₂)
  | TM2.Stmt.goto l => goto fun _ s ↦ normal (l s)
  | TM2.Stmt.halt => halt


theorem trNormal_run {k : K} (s : StAct₂ k) (q : Stmt₂) :
    trNormal (stRun s q) = goto fun _ _ ↦ go k s q := by
  /-
    K : Type u_1
    Γ : K → Type u_2
    Λ : Type u_3
    σ : Type u_4
    k : K
    s : Turing.TM2to1.StAct k
    q : Turing.TM2.Stmt Γ Λ σ
    ⊢ Eq (Turing.TM2to1.trNormal (Turing.TM2to1.stRun s q)) (Turing.TM1.Stmt.goto  …
  -/
              /-
                🎉 no goals
              -/
              /-
                🎉 no goals
              -/
  cases s <;> rfl
              /-
                🎉 no goals
              -/


/-- The set of machine states accessible from an initial TM2 statement. -/
noncomputable def trStmts₁ : Stmt₂ → Finset Λ'₂₁
  | TM2.Stmt.push k f q => {go k (StAct.push f) q, ret q} ∪ trStmts₁ q
  | TM2.Stmt.peek k f q => {go k (StAct.peek f) q, ret q} ∪ trStmts₁ q
  | TM2.Stmt.pop k f q => {go k (StAct.pop f) q, ret q} ∪ trStmts₁ q
  | TM2.Stmt.load _ q => trStmts₁ q
  | TM2.Stmt.branch _ q₁ q₂ => trStmts₁ q₁ ∪ trStmts₁ q₂
  | _ => ∅


theorem trStmts₁_run {k : K} {s : StAct₂ k} {q : Stmt₂} :
    trStmts₁ (stRun s q) = {go k s q, ret q} ∪ trStmts₁ q := by
  /-
    K : Type u_1
    Γ : K → Type u_2
    Λ : Type u_3
    σ : Type u_4
    k : K
    s : Turing.TM2to1.StAct k
    q : Turing.TM2.Stmt Γ Λ σ
    ⊢ Eq (Turing.TM2to1.trStmts₁ (Turing.TM2to1.stRun s q)) (Union.union (Insert.i …
  -/
              /-
                🎉 no goals
              -/
              /-
                🎉 no goals
              -/
  cases s <;> simp only [trStmts₁]
              /-
                🎉 no goals
              -/


theorem tr_respects_aux₂ [DecidableEq K] {k : K} {q : Stmt₂₁} {v : σ} {S : ∀ k, List (Γ k)}
    {L : ListBlank (∀ k, Option (Γ k))}
    (hL : ∀ k, L.map (proj k) = ListBlank.mk ((S k).map some).reverse) (o : StAct₂ k) :
    let v' := stVar v (S k) o
    let Sk' := stWrite v (S k) o
    let S' := update S k Sk'
    ∃ L' : ListBlank (∀ k, Option (Γ k)),
      (∀ k, L'.map (proj k) = ListBlank.mk ((S' k).map some).reverse) ∧
        TM1.stepAux (trStAct q o) v
            ((Tape.move Dir.right)^[(S k).length] (Tape.mk' ∅ (addBottom L))) =
          TM1.stepAux q v' ((Tape.move Dir.right)^[(S' k).length] (Tape.mk' ∅ (addBottom L'))) := by
  /-
    K : Type u_1
    Γ : K → Type u_2
    Λ : Type u_3
    σ : Type u_4
    inst✝ : DecidableEq K
    k : K
    q : Turing.TM1.Stmt Turing.TM2to1.Γ' Turing.TM2to1.Λ' σ
    v : σ
    S : (k : K) → List (Γ k)
    L : Turing.ListBlank ((k : K) → Option (Γ k))
    hL : ∀ (k : K), Eq (Turing.ListBlank.map (Turing.proj k) L) (Turing.ListBlank. …
    o : Turing.TM2to1.StAct k
    ⊢ let v' := Turing.TM2to1.stVar v (S k) o;
      let Sk' := Turing.TM2to1.stWrite v (S k) o;
      let S' := Function.update S k Sk';
      Exists fun L' => And (∀ (k : K), Eq (Turing.ListBlank.map (Turing.proj k) L' …
  -/
  simp only [Function.update_self]; cases o with simp only [stWrite, stVar, trStAct, TM1.stepAux]
  | push f =>
    have := Tape.write_move_right_n fun a : Γ' ↦ (a.1, update a.2 k (some (f v)))
    refine
      ⟨_, fun k' ↦ ?_, by
        -- Porting note: `rw [...]` to `erw [...]; rfl`.
        -- https://github.com/leanprover-community/mathlib4/issues/5164
        erw [Tape.move_right_n_head, List.length, Tape.mk'_nth_nat, this,
          addBottom_modifyNth fun a ↦ update a k (some (f v)), Nat.add_one, iterate_succ']
        rfl⟩
    refine ListBlank.ext fun i ↦ ?_
    rw [ListBlank.nth_map, ListBlank.nth_modifyNth, proj, PointedMap.mk_val]
    by_cases h' : k' = k
    · subst k'
      split_ifs with h
        <;> simp only [List.reverse_cons, Function.update_self, ListBlank.nth_mk, List.map]
      · rw [List.getI_eq_getElem _, List.getElem_append_right] <;>
        simp only [List.length_append, List.length_reverse, List.length_map, ← h,
          Nat.sub_self, List.length_singleton, List.getElem_singleton,
          le_refl, Nat.lt_succ_self]
      rw [← proj_map_nth, hL, ListBlank.nth_mk]
      cases' lt_or_gt_of_ne h with h h
      · rw [List.getI_append]
        simpa only [List.length_map, List.length_reverse] using h
      · rw [gt_iff_lt] at h
        rw [List.getI_eq_default, List.getI_eq_default] <;>
          simp only [Nat.add_one_le_iff, h, List.length, le_of_lt, List.length_reverse,
            List.length_append, List.length_map]
    · split_ifs <;> rw [Function.update_of_ne h', ← proj_map_nth, hL]
      rw [Function.update_of_ne h']
  | peek f =>
    rw [Function.update_eq_self]
    use L, hL; rw [Tape.move_left_right]; congr
    cases e : S k; · rfl
    rw [List.length_cons, iterate_succ', Function.comp, Tape.move_right_left,
      Tape.move_right_n_head, Tape.mk'_nth_nat, addBottom_nth_snd, stk_nth_val _ (hL k), e,
      List.reverse_cons, ← List.length_reverse, List.getElem?_concat_length]
    rfl
  | pop f =>
    cases' e : S k with hd tl
    · simp only [Tape.mk'_head, ListBlank.head_cons, Tape.move_left_mk', List.length,
        Tape.write_mk', List.head?, iterate_zero_apply, List.tail_nil]
      rw [← e, Function.update_eq_self]
      exact ⟨L, hL, by rw [addBottom_head_fst, cond]⟩
    · refine
        ⟨_, fun k' ↦ ?_, by
          erw [List.length_cons, Tape.move_right_n_head, Tape.mk'_nth_nat, addBottom_nth_succ_fst,
            cond_false, iterate_succ', Function.comp, Tape.move_right_left, Tape.move_right_n_head,
            Tape.mk'_nth_nat, Tape.write_move_right_n fun a : Γ' ↦ (a.1, update a.2 k none),
            addBottom_modifyNth fun a ↦ update a k none, addBottom_nth_snd,
            stk_nth_val _ (hL k), e,
            show (List.cons hd tl).reverse[tl.length]? = some hd by
              rw [List.reverse_cons, ← List.length_reverse, List.getElem?_concat_length],
            List.head?, List.tail]⟩
      refine ListBlank.ext fun i ↦ ?_
      rw [ListBlank.nth_map, ListBlank.nth_modifyNth, proj, PointedMap.mk_val]
      by_cases h' : k' = k
      · subst k'
        split_ifs with h <;> simp only [Function.update_self, ListBlank.nth_mk, List.tail]
        · rw [List.getI_eq_default]
          · rfl
          rw [h, List.length_reverse, List.length_map]
        rw [← proj_map_nth, hL, ListBlank.nth_mk, e, List.map, List.reverse_cons]
        cases' lt_or_gt_of_ne h with h h
        · rw [List.getI_append]
          simpa only [List.length_map, List.length_reverse] using h
        · rw [gt_iff_lt] at h
          rw [List.getI_eq_default, List.getI_eq_default] <;>
            simp only [Nat.add_one_le_iff, h, List.length, le_of_lt, List.length_reverse,
              List.length_append, List.length_map]
      · split_ifs <;> rw [Function.update_of_ne h', ← proj_map_nth, hL]
        rw [Function.update_of_ne h']


/-- The TM2 emulator machine states written as a TM1 program.
This handles the `go` and `ret` states, which shuttle to and from a stack top. -/
def tr : Λ'₂₁ → Stmt₂₁
  | normal q => trNormal (M q)
  | go k s q =>
    branch (fun a _ ↦ (a.2 k).isNone) (trStAct (goto fun _ _ ↦ ret q) s)
      (move Dir.right <| goto fun _ _ ↦ go k s q)
  | ret q => branch (fun a _ ↦ a.1) (trNormal q) (move Dir.left <| goto fun _ _ ↦ ret q)

-- Porting note: unknown attribute
-- attribute [local pp_using_anonymous_constructor] Turing.TM1.Cfg


/-- The relation between TM2 configurations and TM1 configurations of the TM2 emulator. -/
inductive TrCfg : Cfg₂ → Cfg₂₁ → Prop
  | mk {q : Option Λ} {v : σ} {S : ∀ k, List (Γ k)} (L : ListBlank (∀ k, Option (Γ k))) :
    (∀ k, L.map (proj k) = ListBlank.mk ((S k).map some).reverse) →
      TrCfg ⟨q, v, S⟩ ⟨q.map normal, v, Tape.mk' ∅ (addBottom L)⟩


theorem tr_respects_aux₁ {k} (o q v) {S : List (Γ k)} {L : ListBlank (∀ k, Option (Γ k))}
    (hL : L.map (proj k) = ListBlank.mk (S.map some).reverse) (n) (H : n ≤ S.length) :
    Reaches₀ (TM1.step (tr M)) ⟨some (go k o q), v, Tape.mk' ∅ (addBottom L)⟩
      ⟨some (go k o q), v, (Tape.move Dir.right)^[n] (Tape.mk' ∅ (addBottom L))⟩ := by
  /-
    K : Type u_1
    Γ : K → Type u_2
    Λ : Type u_3
    σ : Type u_4
    inst✝ : DecidableEq K
    M : Λ → Turing.TM2.Stmt Γ Λ σ
    k : K
    o : Turing.TM2to1.StAct k
    q : Turing.TM2.Stmt Γ Λ σ
    v : σ
    S : List (Γ k)
    L : Turing.ListBlank ((k : K) → Option (Γ k))
    hL : Eq (Turing.ListBlank.map (Turing.proj k) L) (Turing.ListBlank.mk (List.ma …
    n : Nat
    H : LE.le n S.length
    ⊢ Turing.Reaches₀ (Turing.TM1.step (Turing.TM2to1.tr M)) { l := Option.some (T …
  -/
  induction' n with n IH; · rfl
                            /-
                              🎉 no goals
                            -/
  /-
    case succ
    K : Type u_1
    Γ : K → Type u_2
    Λ : Type u_3
    σ : Type u_4
    inst✝ : DecidableEq K
    M : Λ → Turing.TM2.Stmt Γ Λ σ
    k : K
    o : Turing.TM2to1.StAct k
    q : Turing.TM2.Stmt Γ Λ σ
    v : σ
    S : List (Γ k)
    L : Turing.ListBlank ((k : K) → Option (Γ k))
    hL : Eq (Turing.ListBlank.map (Turing.proj k) L) (Turing.ListBlank.mk (List.ma …
    n : Nat
    IH : LE.le n S.length → Turing.Reaches₀ (Turing.TM1.step (Turing.TM2to1.tr M)) …
    H : LE.le (HAdd.hAdd n 1) S.length
    ⊢ Turing.Reaches₀ (Turing.TM1.step (Turing.TM2to1.tr M)) { l := Option.some (T …
  -/
  apply (IH (le_of_lt H)).tail
  /-
    case succ
    K : Type u_1
    Γ : K → Type u_2
    Λ : Type u_3
    σ : Type u_4
    inst✝ : DecidableEq K
    M : Λ → Turing.TM2.Stmt Γ Λ σ
    k : K
    o : Turing.TM2to1.StAct k
    q : Turing.TM2.Stmt Γ Λ σ
    v : σ
    S : List (Γ k)
    L : Turing.ListBlank ((k : K) → Option (Γ k))
    hL : Eq (Turing.ListBlank.map (Turing.proj k) L) (Turing.ListBlank.mk (List.ma …
    n : Nat
    IH : LE.le n S.length → Turing.Reaches₀ (Turing.TM1.step (Turing.TM2to1.tr M)) …
    H : LE.le (HAdd.hAdd n 1) S.length
    ⊢ Membership.mem (Turing.TM1.step (Turing.TM2to1.tr M) { l := Option.some (Tur …
  -/
  rw [iterate_succ_apply']
  simp only [TM1.step, TM1.stepAux, tr, Tape.mk'_nth_nat, Tape.move_right_n_head,
    addBottom_nth_snd, Option.mem_def]
  /-
    case succ
    K : Type u_1
    Γ : K → Type u_2
    Λ : Type u_3
    σ : Type u_4
    inst✝ : DecidableEq K
    M : Λ → Turing.TM2.Stmt Γ Λ σ
    k : K
    o : Turing.TM2to1.StAct k
    q : Turing.TM2.Stmt Γ Λ σ
    v : σ
    S : List (Γ k)
    L : Turing.ListBlank ((k : K) → Option (Γ k))
    hL : Eq (Turing.ListBlank.map (Turing.proj k) L) (Turing.ListBlank.mk (List.ma …
    n : Nat
    IH : LE.le n S.length → Turing.Reaches₀ (Turing.TM1.step (Turing.TM2to1.tr M)) …
    H : LE.le (HAdd.hAdd n 1) S.length
    ⊢ Eq (Option.some (cond (L.nth n k).isNone (Turing.TM1.stepAux (Turing.TM2to1. …
  -/
  rw [stk_nth_val _ hL, List.getElem?_eq_getElem]
    /-
      case succ
      K : Type u_1
      Γ : K → Type u_2
      Λ : Type u_3
      σ : Type u_4
      inst✝ : DecidableEq K
      M : Λ → Turing.TM2.Stmt Γ Λ σ
      k : K
      o : Turing.TM2to1.StAct k
      q : Turing.TM2.Stmt Γ Λ σ
      v : σ
      S : List (Γ k)
      L : Turing.ListBlank ((k : K) → Option (Γ k))
      hL : Eq (Turing.ListBlank.map (Turing.proj k) L) (Turing.ListBlank.mk (List.ma …
      n : Nat
      IH : LE.le n S.length → Turing.Reaches₀ (Turing.TM1.step (Turing.TM2to1.tr M)) …
      H : LE.le (HAdd.hAdd n 1) S.length
      ⊢ Eq (Option.some (cond (Option.some (GetElem.getElem S.reverse n ?succ)).isNo …
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case succ
      K : Type u_1
      Γ : K → Type u_2
      Λ : Type u_3
      σ : Type u_4
      inst✝ : DecidableEq K
      M : Λ → Turing.TM2.Stmt Γ Λ σ
      k : K
      o : Turing.TM2to1.StAct k
      q : Turing.TM2.Stmt Γ Λ σ
      v : σ
      S : List (Γ k)
      L : Turing.ListBlank ((k : K) → Option (Γ k))
      hL : Eq (Turing.ListBlank.map (Turing.proj k) L) (Turing.ListBlank.mk (List.ma …
      n : Nat
      IH : LE.le n S.length → Turing.Reaches₀ (Turing.TM1.step (Turing.TM2to1.tr M)) …
      H : LE.le (HAdd.hAdd n 1) S.length
      ⊢ LT.lt n S.reverse.length
    -/
  · rwa [List.length_reverse]
    /-
      🎉 no goals
    -/


theorem tr_respects_aux₃ {q v} {L : ListBlank (∀ k, Option (Γ k))} (n) : Reaches₀ (TM1.step (tr M))
    ⟨some (ret q), v, (Tape.move Dir.right)^[n] (Tape.mk' ∅ (addBottom L))⟩
    ⟨some (ret q), v, Tape.mk' ∅ (addBottom L)⟩ := by
  /-
    K : Type u_1
    Γ : K → Type u_2
    Λ : Type u_3
    σ : Type u_4
    inst✝ : DecidableEq K
    M : Λ → Turing.TM2.Stmt Γ Λ σ
    q : Turing.TM2.Stmt Γ Λ σ
    v : σ
    L : Turing.ListBlank ((k : K) → Option (Γ k))
    n : Nat
    ⊢ Turing.Reaches₀ (Turing.TM1.step (Turing.TM2to1.tr M)) { l := Option.some (T …
  -/
  induction' n with n IH; · rfl
                            /-
                              🎉 no goals
                            -/
  /-
    case succ
    K : Type u_1
    Γ : K → Type u_2
    Λ : Type u_3
    σ : Type u_4
    inst✝ : DecidableEq K
    M : Λ → Turing.TM2.Stmt Γ Λ σ
    q : Turing.TM2.Stmt Γ Λ σ
    v : σ
    L : Turing.ListBlank ((k : K) → Option (Γ k))
    n : Nat
    IH : Turing.Reaches₀ (Turing.TM1.step (Turing.TM2to1.tr M)) { l := Option.some …
    ⊢ Turing.Reaches₀ (Turing.TM1.step (Turing.TM2to1.tr M)) { l := Option.some (T …
  -/
  refine Reaches₀.head ?_ IH
  /-
    case succ
    K : Type u_1
    Γ : K → Type u_2
    Λ : Type u_3
    σ : Type u_4
    inst✝ : DecidableEq K
    M : Λ → Turing.TM2.Stmt Γ Λ σ
    q : Turing.TM2.Stmt Γ Λ σ
    v : σ
    L : Turing.ListBlank ((k : K) → Option (Γ k))
    n : Nat
    IH : Turing.Reaches₀ (Turing.TM1.step (Turing.TM2to1.tr M)) { l := Option.some …
    ⊢ Membership.mem (Turing.TM1.step (Turing.TM2to1.tr M) { l := Option.some (Tur …
  -/
  simp only [Option.mem_def, TM1.step]
  rw [Option.some_inj, tr, TM1.stepAux, Tape.move_right_n_head, Tape.mk'_nth_nat,
    addBottom_nth_succ_fst, TM1.stepAux, iterate_succ', Function.comp_apply, Tape.move_right_left]
  /-
    case succ
    K : Type u_1
    Γ : K → Type u_2
    Λ : Type u_3
    σ : Type u_4
    inst✝ : DecidableEq K
    M : Λ → Turing.TM2.Stmt Γ Λ σ
    q : Turing.TM2.Stmt Γ Λ σ
    v : σ
    L : Turing.ListBlank ((k : K) → Option (Γ k))
    n : Nat
    IH : Turing.Reaches₀ (Turing.TM1.step (Turing.TM2to1.tr M)) { l := Option.some …
    ⊢ Eq (cond Bool.false (Turing.TM1.stepAux (Turing.TM2to1.trNormal q) v (Turing …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem tr_respects_aux {q v T k} {S : ∀ k, List (Γ k)}
    (hT : ∀ k, ListBlank.map (proj k) T = ListBlank.mk ((S k).map some).reverse) (o : StAct₂ k)
    (IH : ∀ {v : σ} {S : ∀ k : K, List (Γ k)} {T : ListBlank (∀ k, Option (Γ k))},
      (∀ k, ListBlank.map (proj k) T = ListBlank.mk ((S k).map some).reverse) →
      ∃ b, TrCfg (TM2.stepAux q v S) b ∧
        Reaches (TM1.step (tr M)) (TM1.stepAux (trNormal q) v (Tape.mk' ∅ (addBottom T))) b) :
    ∃ b, TrCfg (TM2.stepAux (stRun o q) v S) b ∧ Reaches (TM1.step (tr M))
      (TM1.stepAux (trNormal (stRun o q)) v (Tape.mk' ∅ (addBottom T))) b := by
  /-
    K : Type u_1
    Γ : K → Type u_2
    Λ : Type u_3
    σ : Type u_4
    inst✝ : DecidableEq K
    M : Λ → Turing.TM2.Stmt Γ Λ σ
    q : Turing.TM2.Stmt Γ Λ σ
    v : σ
    T : Turing.ListBlank ((i : K) → Option (Γ i))
    k : K
    S : (k : K) → List (Γ k)
    hT : ∀ (k : K), Eq (Turing.ListBlank.map (Turing.proj k) T) (Turing.ListBlank. …
    o : Turing.TM2to1.StAct k
    IH : ∀ {v : σ} {S : (k : K) → List (Γ k)} {T : Turing.ListBlank ((k : K) → Opt …
    ⊢ Exists fun b => And (Turing.TM2to1.TrCfg (Turing.TM2.stepAux (Turing.TM2to1. …
  -/
  simp only [trNormal_run, step_run]
  /-
    K : Type u_1
    Γ : K → Type u_2
    Λ : Type u_3
    σ : Type u_4
    inst✝ : DecidableEq K
    M : Λ → Turing.TM2.Stmt Γ Λ σ
    q : Turing.TM2.Stmt Γ Λ σ
    v : σ
    T : Turing.ListBlank ((i : K) → Option (Γ i))
    k : K
    S : (k : K) → List (Γ k)
    hT : ∀ (k : K), Eq (Turing.ListBlank.map (Turing.proj k) T) (Turing.ListBlank. …
    o : Turing.TM2to1.StAct k
    IH : ∀ {v : σ} {S : (k : K) → List (Γ k)} {T : Turing.ListBlank ((k : K) → Opt …
    ⊢ Exists fun b => And (Turing.TM2to1.TrCfg (Turing.TM2.stepAux q (Turing.TM2to …
  -/
  have hgo := tr_respects_aux₁ M o q v (hT k) _ le_rfl
  /-
    K : Type u_1
    Γ : K → Type u_2
    Λ : Type u_3
    σ : Type u_4
    inst✝ : DecidableEq K
    M : Λ → Turing.TM2.Stmt Γ Λ σ
    q : Turing.TM2.Stmt Γ Λ σ
    v : σ
    T : Turing.ListBlank ((i : K) → Option (Γ i))
    k : K
    S : (k : K) → List (Γ k)
    hT : ∀ (k : K), Eq (Turing.ListBlank.map (Turing.proj k) T) (Turing.ListBlank. …
    o : Turing.TM2to1.StAct k
    IH : ∀ {v : σ} {S : (k : K) → List (Γ k)} {T : Turing.ListBlank ((k : K) → Opt …
    hgo : Turing.Reaches₀ (Turing.TM1.step (Turing.TM2to1.tr M)) { l := Option.som …
    ⊢ Exists fun b => And (Turing.TM2to1.TrCfg (Turing.TM2.stepAux q (Turing.TM2to …
  -/
  obtain ⟨T', hT', hrun⟩ := tr_respects_aux₂ (Λ := Λ) hT o
  /-
    case intro.intro
    K : Type u_1
    Γ : K → Type u_2
    Λ : Type u_3
    σ : Type u_4
    inst✝ : DecidableEq K
    M : Λ → Turing.TM2.Stmt Γ Λ σ
    q : Turing.TM2.Stmt Γ Λ σ
    v : σ
    T : Turing.ListBlank ((i : K) → Option (Γ i))
    k : K
    S : (k : K) → List (Γ k)
    hT : ∀ (k : K), Eq (Turing.ListBlank.map (Turing.proj k) T) (Turing.ListBlank. …
    o : Turing.TM2to1.StAct k
    IH : ∀ {v : σ} {S : (k : K) → List (Γ k)} {T : Turing.ListBlank ((k : K) → Opt …
    hgo : Turing.Reaches₀ (Turing.TM1.step (Turing.TM2to1.tr M)) { l := Option.som …
    T' : Turing.ListBlank ((k : K) → Option (Γ k))
    hT' : ∀ (k_1 : K), Eq (Turing.ListBlank.map (Turing.proj k_1) T') (Turing.List …
    hrun : Eq (Turing.TM1.stepAux (Turing.TM2to1.trStAct ?m.395379 o) ?m.395380 (N …
    ⊢ Exists fun b => And (Turing.TM2to1.TrCfg (Turing.TM2.stepAux q (Turing.TM2to …
  -/
  have := hgo.tail' rfl
  rw [tr, TM1.stepAux, Tape.move_right_n_head, Tape.mk'_nth_nat, addBottom_nth_snd,
    stk_nth_val _ (hT k), List.getElem?_eq_none (le_of_eq (List.length_reverse _)),
    Option.isNone, cond, hrun, TM1.stepAux] at this
  /-
    case intro.intro
    K : Type u_1
    Γ : K → Type u_2
    Λ : Type u_3
    σ : Type u_4
    inst✝ : DecidableEq K
    M : Λ → Turing.TM2.Stmt Γ Λ σ
    q : Turing.TM2.Stmt Γ Λ σ
    v : σ
    T : Turing.ListBlank ((i : K) → Option (Γ i))
    k : K
    S : (k : K) → List (Γ k)
    hT : ∀ (k : K), Eq (Turing.ListBlank.map (Turing.proj k) T) (Turing.ListBlank. …
    o : Turing.TM2to1.StAct k
    IH : ∀ {v : σ} {S : (k : K) → List (Γ k)} {T : Turing.ListBlank ((k : K) → Opt …
    hgo : Turing.Reaches₀ (Turing.TM1.step (Turing.TM2to1.tr M)) { l := Option.som …
    T' : Turing.ListBlank ((k : K) → Option (Γ k))
    hT' : ∀ (k_1 : K), Eq (Turing.ListBlank.map (Turing.proj k_1) T') (Turing.List …
    hrun : Eq (Turing.TM1.stepAux (Turing.TM2to1.trStAct (Turing.TM1.Stmt.goto fun …
    this : Turing.Reaches₁ (Turing.TM1.step (Turing.TM2to1.tr M)) { l := Option.so …
    ⊢ Exists fun b => And (Turing.TM2to1.TrCfg (Turing.TM2.stepAux q (Turing.TM2to …
  -/
  obtain ⟨c, gc, rc⟩ := IH hT'
  /-
    case intro.intro.intro.intro
    K : Type u_1
    Γ : K → Type u_2
    Λ : Type u_3
    σ : Type u_4
    inst✝ : DecidableEq K
    M : Λ → Turing.TM2.Stmt Γ Λ σ
    q : Turing.TM2.Stmt Γ Λ σ
    v : σ
    T : Turing.ListBlank ((i : K) → Option (Γ i))
    k : K
    S : (k : K) → List (Γ k)
    hT : ∀ (k : K), Eq (Turing.ListBlank.map (Turing.proj k) T) (Turing.ListBlank. …
    o : Turing.TM2to1.StAct k
    IH : ∀ {v : σ} {S : (k : K) → List (Γ k)} {T : Turing.ListBlank ((k : K) → Opt …
    hgo : Turing.Reaches₀ (Turing.TM1.step (Turing.TM2to1.tr M)) { l := Option.som …
    T' : Turing.ListBlank ((k : K) → Option (Γ k))
    hT' : ∀ (k_1 : K), Eq (Turing.ListBlank.map (Turing.proj k_1) T') (Turing.List …
    hrun : Eq (Turing.TM1.stepAux (Turing.TM2to1.trStAct (Turing.TM1.Stmt.goto fun …
    this : Turing.Reaches₁ (Turing.TM1.step (Turing.TM2to1.tr M)) { l := Option.so …
    c : Turing.TM1.Cfg Turing.TM2to1.Γ' Turing.TM2to1.Λ' σ
    gc : Turing.TM2to1.TrCfg (Turing.TM2.stepAux q ?m.401048 (Function.update S k  …
    rc : Turing.Reaches (Turing.TM1.step (Turing.TM2to1.tr M)) (Turing.TM1.stepAux …
    ⊢ Exists fun b => And (Turing.TM2to1.TrCfg (Turing.TM2.stepAux q (Turing.TM2to …
  -/
  refine ⟨c, gc, (this.to₀.trans (tr_respects_aux₃ M _) c (TransGen.head' rfl ?_)).to_reflTransGen⟩
  /-
    case intro.intro.intro.intro
    K : Type u_1
    Γ : K → Type u_2
    Λ : Type u_3
    σ : Type u_4
    inst✝ : DecidableEq K
    M : Λ → Turing.TM2.Stmt Γ Λ σ
    q : Turing.TM2.Stmt Γ Λ σ
    v : σ
    T : Turing.ListBlank ((i : K) → Option (Γ i))
    k : K
    S : (k : K) → List (Γ k)
    hT : ∀ (k : K), Eq (Turing.ListBlank.map (Turing.proj k) T) (Turing.ListBlank. …
    o : Turing.TM2to1.StAct k
    IH : ∀ {v : σ} {S : (k : K) → List (Γ k)} {T : Turing.ListBlank ((k : K) → Opt …
    hgo : Turing.Reaches₀ (Turing.TM1.step (Turing.TM2to1.tr M)) { l := Option.som …
    T' : Turing.ListBlank ((k : K) → Option (Γ k))
    hT' : ∀ (k_1 : K), Eq (Turing.ListBlank.map (Turing.proj k_1) T') (Turing.List …
    hrun : Eq (Turing.TM1.stepAux (Turing.TM2to1.trStAct (Turing.TM1.Stmt.goto fun …
    this : Turing.Reaches₁ (Turing.TM1.step (Turing.TM2to1.tr M)) { l := Option.so …
    c : Turing.TM1.Cfg Turing.TM2to1.Γ' Turing.TM2to1.Λ' σ
    gc : Turing.TM2to1.TrCfg (Turing.TM2.stepAux q (Turing.TM2to1.stVar v (S k) o) …
    rc : Turing.Reaches (Turing.TM1.step (Turing.TM2to1.tr M)) (Turing.TM1.stepAux …
    ⊢ Relation.ReflTransGen (fun a b => Membership.mem (Turing.TM1.step (Turing.TM …
  -/
  rw [tr, TM1.stepAux, Tape.mk'_head, addBottom_head_fst]
  /-
    case intro.intro.intro.intro
    K : Type u_1
    Γ : K → Type u_2
    Λ : Type u_3
    σ : Type u_4
    inst✝ : DecidableEq K
    M : Λ → Turing.TM2.Stmt Γ Λ σ
    q : Turing.TM2.Stmt Γ Λ σ
    v : σ
    T : Turing.ListBlank ((i : K) → Option (Γ i))
    k : K
    S : (k : K) → List (Γ k)
    hT : ∀ (k : K), Eq (Turing.ListBlank.map (Turing.proj k) T) (Turing.ListBlank. …
    o : Turing.TM2to1.StAct k
    IH : ∀ {v : σ} {S : (k : K) → List (Γ k)} {T : Turing.ListBlank ((k : K) → Opt …
    hgo : Turing.Reaches₀ (Turing.TM1.step (Turing.TM2to1.tr M)) { l := Option.som …
    T' : Turing.ListBlank ((k : K) → Option (Γ k))
    hT' : ∀ (k_1 : K), Eq (Turing.ListBlank.map (Turing.proj k_1) T') (Turing.List …
    hrun : Eq (Turing.TM1.stepAux (Turing.TM2to1.trStAct (Turing.TM1.Stmt.goto fun …
    this : Turing.Reaches₁ (Turing.TM1.step (Turing.TM2to1.tr M)) { l := Option.so …
    c : Turing.TM1.Cfg Turing.TM2to1.Γ' Turing.TM2to1.Λ' σ
    gc : Turing.TM2to1.TrCfg (Turing.TM2.stepAux q (Turing.TM2to1.stVar v (S k) o) …
    rc : Turing.Reaches (Turing.TM1.step (Turing.TM2to1.tr M)) (Turing.TM1.stepAux …
    ⊢ Relation.ReflTransGen (fun a b => Membership.mem (Turing.TM1.step (Turing.TM …
  -/
  exact rc
  /-
    🎉 no goals
  -/


attribute [local simp] Respects TM2.step TM2.stepAux trNormal


theorem tr_respects : Respects (TM2.step M) (TM1.step (tr M)) TrCfg := by
  -- Porting note (https://github.com/leanprover-community/mathlib4/issues/12129): additional beta reduction needed
  /-
    K : Type u_1
    Γ : K → Type u_2
    Λ : Type u_3
    σ : Type u_4
    inst✝ : DecidableEq K
    M : Λ → Turing.TM2.Stmt Γ Λ σ
    ⊢ Turing.Respects (Turing.TM2.step M) (Turing.TM1.step (Turing.TM2to1.tr M)) T …
  -/
  intro c₁ c₂ h
  /-
    K : Type u_1
    Γ : K → Type u_2
    Λ : Type u_3
    σ : Type u_4
    inst✝ : DecidableEq K
    M : Λ → Turing.TM2.Stmt Γ Λ σ
    c₁ : Turing.TM2.Cfg Γ Λ σ
    c₂ : Turing.TM1.Cfg Turing.TM2to1.Γ' Turing.TM2to1.Λ' σ
    h : Turing.TM2to1.TrCfg c₁ c₂
    ⊢ Turing.Respects.match_1 (fun x => Prop) (Turing.TM2.step M c₁) (fun b₁ => Ex …
  -/
  cases' h with l v S L hT
  /-
    case mk
    K : Type u_1
    Γ : K → Type u_2
    Λ : Type u_3
    σ : Type u_4
    inst✝ : DecidableEq K
    M : Λ → Turing.TM2.Stmt Γ Λ σ
    l : Option Λ
    v : σ
    S : (k : K) → List (Γ k)
    L : Turing.ListBlank ((k : K) → Option (Γ k))
    hT : ∀ (k : K), Eq (Turing.ListBlank.map (Turing.proj k) L) (Turing.ListBlank. …
    ⊢ Turing.Respects.match_1 (fun x => Prop) (Turing.TM2.step M { l := l, var :=  …
  -/
  cases' l with l; · constructor
                     /-
                       🎉 no goals
                     -/
  /-
    case mk.some
    K : Type u_1
    Γ : K → Type u_2
    Λ : Type u_3
    σ : Type u_4
    inst✝ : DecidableEq K
    M : Λ → Turing.TM2.Stmt Γ Λ σ
    v : σ
    S : (k : K) → List (Γ k)
    L : Turing.ListBlank ((k : K) → Option (Γ k))
    hT : ∀ (k : K), Eq (Turing.ListBlank.map (Turing.proj k) L) (Turing.ListBlank. …
    l : Λ
    ⊢ Turing.Respects.match_1 (fun x => Prop) (Turing.TM2.step M { l := Option.som …
  -/
  rsuffices ⟨b, c, r⟩ : ∃ b, _ ∧ Reaches (TM1.step (tr M)) _ _
    /-
      case mk.some.intro.intro
      K : Type u_1
      Γ : K → Type u_2
      Λ : Type u_3
      σ : Type u_4
      inst✝ : DecidableEq K
      M : Λ → Turing.TM2.Stmt Γ Λ σ
      v : σ
      S : (k : K) → List (Γ k)
      L : Turing.ListBlank ((k : K) → Option (Γ k))
      hT : ∀ (k : K), Eq (Turing.ListBlank.map (Turing.proj k) L) (Turing.ListBlank. …
      l : Λ
      b : ?m.406128
      c : ?m.406454 b
      r : Turing.Reaches (Turing.TM1.step (Turing.TM2to1.tr M)) (?m.406455 b) (?m.40 …
      ⊢ Turing.Respects.match_1 (fun x => Prop) (Turing.TM2.step M { l := Option.som …
    -/
  · exact ⟨b, c, TransGen.head' rfl r⟩
    /-
      🎉 no goals
    -/
  /-
    K : Type u_1
    Γ : K → Type u_2
    Λ : Type u_3
    σ : Type u_4
    inst✝ : DecidableEq K
    M : Λ → Turing.TM2.Stmt Γ Λ σ
    v : σ
    S : (k : K) → List (Γ k)
    L : Turing.ListBlank ((k : K) → Option (Γ k))
    hT : ∀ (k : K), Eq (Turing.ListBlank.map (Turing.proj k) L) (Turing.ListBlank. …
    l : Λ
    ⊢ Exists fun b => And (Turing.TM2to1.TrCfg (Turing.TM2.stepAux (M l) v S) b) ( …
  -/
  simp only [tr]
  -- Porting note: `refine'` failed because of implicit lambda, so `induction` is used.
  /-
    K : Type u_1
    Γ : K → Type u_2
    Λ : Type u_3
    σ : Type u_4
    inst✝ : DecidableEq K
    M : Λ → Turing.TM2.Stmt Γ Λ σ
    v : σ
    S : (k : K) → List (Γ k)
    L : Turing.ListBlank ((k : K) → Option (Γ k))
    hT : ∀ (k : K), Eq (Turing.ListBlank.map (Turing.proj k) L) (Turing.ListBlank. …
    l : Λ
    ⊢ Exists fun b => And (Turing.TM2to1.TrCfg (Turing.TM2.stepAux (M l) v S) b) ( …
  -/
  generalize M l = N
  induction N using stmtStRec generalizing v S L hT with
  | H₁ k s q IH => exact tr_respects_aux M hT s @IH
  | H₂ a _ IH => exact IH _ hT
  | H₃ p q₁ q₂ IH₁ IH₂ =>
    unfold TM2.stepAux trNormal TM1.stepAux
    beta_reduce
    cases p v <;> [exact IH₂ _ hT; exact IH₁ _ hT]
  | H₄ => exact ⟨_, ⟨_, hT⟩, ReflTransGen.refl⟩
  | H₅ => exact ⟨_, ⟨_, hT⟩, ReflTransGen.refl⟩


theorem trCfg_init (k) (L : List (Γ k)) : TrCfg (TM2.init k L) (TM1.init (trInit k L) : Cfg₂₁) := by
  /-
    K : Type u_1
    Γ : K → Type u_2
    Λ : Type u_3
    σ : Type u_4
    inst✝² : DecidableEq K
    inst✝¹ : Inhabited Λ
    inst✝ : Inhabited σ
    k : K
    L : List (Γ k)
    ⊢ Turing.TM2to1.TrCfg (Turing.TM2.init k L) (Turing.TM1.init (Turing.TM2to1.tr …
  -/
  rw [(_ : TM1.init _ = _)]
    /-
      K : Type u_1
      Γ : K → Type u_2
      Λ : Type u_3
      σ : Type u_4
      inst✝² : DecidableEq K
      inst✝¹ : Inhabited Λ
      inst✝ : Inhabited σ
      k : K
      L : List (Γ k)
      ⊢ Turing.TM2to1.TrCfg (Turing.TM2.init k L) ?m.415451
    -/
  · refine ⟨ListBlank.mk (L.reverse.map fun a ↦ update default k (some a)), fun k' ↦ ?_⟩
    /-
      K : Type u_1
      Γ : K → Type u_2
      Λ : Type u_3
      σ : Type u_4
      inst✝² : DecidableEq K
      inst✝¹ : Inhabited Λ
      inst✝ : Inhabited σ
      k : K
      L : List (Γ k)
      k' : K
      ⊢ Eq (Turing.ListBlank.map (Turing.proj k') (Turing.ListBlank.mk (List.map (fu …
    -/
    refine ListBlank.ext fun i ↦ ?_
    /-
      K : Type u_1
      Γ : K → Type u_2
      Λ : Type u_3
      σ : Type u_4
      inst✝² : DecidableEq K
      inst✝¹ : Inhabited Λ
      inst✝ : Inhabited σ
      k : K
      L : List (Γ k)
      k' : K
      i : Nat
      ⊢ Eq ((Turing.ListBlank.map (Turing.proj k') (Turing.ListBlank.mk (List.map (f …
    -/
    rw [ListBlank.map_mk, ListBlank.nth_mk, List.getI_eq_iget_getElem?, List.map_map]
    have : ((proj k').f ∘ fun a => update (β := fun k => Option (Γ k)) default k (some a))
      = fun a => (proj k').f (update (β := fun k => Option (Γ k)) default k (some a)) := rfl
    /-
      K : Type u_1
      Γ : K → Type u_2
      Λ : Type u_3
      σ : Type u_4
      inst✝² : DecidableEq K
      inst✝¹ : Inhabited Λ
      inst✝ : Inhabited σ
      k : K
      L : List (Γ k)
      k' : K
      i : Nat
      this : Eq (Function.comp (Turing.proj k').f fun a => Function.update Inhabited …
      ⊢ Eq (GetElem?.getElem? (List.map (Function.comp (Turing.proj k').f fun a => F …
    -/
    rw [this, List.getElem?_map, proj, PointedMap.mk_val]
    /-
      K : Type u_1
      Γ : K → Type u_2
      Λ : Type u_3
      σ : Type u_4
      inst✝² : DecidableEq K
      inst✝¹ : Inhabited Λ
      inst✝ : Inhabited σ
      k : K
      L : List (Γ k)
      k' : K
      i : Nat
      this : Eq (Function.comp (Turing.proj k').f fun a => Function.update Inhabited …
      ⊢ Eq (Option.map (fun a => Function.update Inhabited.default k (Option.some a) …
    -/
    simp only []
    /-
      K : Type u_1
      Γ : K → Type u_2
      Λ : Type u_3
      σ : Type u_4
      inst✝² : DecidableEq K
      inst✝¹ : Inhabited Λ
      inst✝ : Inhabited σ
      k : K
      L : List (Γ k)
      k' : K
      i : Nat
      this : Eq (Function.comp (Turing.proj k').f fun a => Function.update Inhabited …
      ⊢ Eq (Option.map (fun a => Function.update Inhabited.default k (Option.some a) …
    -/
    by_cases h : k' = k
      /-
        case pos
        K : Type u_1
        Γ : K → Type u_2
        Λ : Type u_3
        σ : Type u_4
        inst✝² : DecidableEq K
        inst✝¹ : Inhabited Λ
        inst✝ : Inhabited σ
        k : K
        L : List (Γ k)
        k' : K
        i : Nat
        this : Eq (Function.comp (Turing.proj k').f fun a => Function.update Inhabited …
        h : Eq k' k
        ⊢ Eq (Option.map (fun a => Function.update Inhabited.default k (Option.some a) …
      -/
    · subst k'
      /-
        case pos
        K : Type u_1
        Γ : K → Type u_2
        Λ : Type u_3
        σ : Type u_4
        inst✝² : DecidableEq K
        inst✝¹ : Inhabited Λ
        inst✝ : Inhabited σ
        k : K
        L : List (Γ k)
        i : Nat
        this : Eq (Function.comp (Turing.proj k).f fun a => Function.update Inhabited. …
        ⊢ Eq (Option.map (fun a => Function.update Inhabited.default k (Option.some a) …
      -/
      simp only [Function.update_self]
      /-
        case pos
        K : Type u_1
        Γ : K → Type u_2
        Λ : Type u_3
        σ : Type u_4
        inst✝² : DecidableEq K
        inst✝¹ : Inhabited Λ
        inst✝ : Inhabited σ
        k : K
        L : List (Γ k)
        i : Nat
        this : Eq (Function.comp (Turing.proj k).f fun a => Function.update Inhabited. …
        ⊢ Eq (Option.map (fun a => Option.some a) (GetElem?.getElem? L.reverse i)).ige …
      -/
      rw [ListBlank.nth_mk, List.getI_eq_iget_getElem?, ← List.map_reverse, List.getElem?_map]
      /-
        🎉 no goals
      -/
      /-
        case neg
        K : Type u_1
        Γ : K → Type u_2
        Λ : Type u_3
        σ : Type u_4
        inst✝² : DecidableEq K
        inst✝¹ : Inhabited Λ
        inst✝ : Inhabited σ
        k : K
        L : List (Γ k)
        k' : K
        i : Nat
        this : Eq (Function.comp (Turing.proj k').f fun a => Function.update Inhabited …
        h : Not (Eq k' k)
        ⊢ Eq (Option.map (fun a => Function.update Inhabited.default k (Option.some a) …
      -/
    · simp only [Function.update_of_ne h]
      /-
        case neg
        K : Type u_1
        Γ : K → Type u_2
        Λ : Type u_3
        σ : Type u_4
        inst✝² : DecidableEq K
        inst✝¹ : Inhabited Λ
        inst✝ : Inhabited σ
        k : K
        L : List (Γ k)
        k' : K
        i : Nat
        this : Eq (Function.comp (Turing.proj k').f fun a => Function.update Inhabited …
        h : Not (Eq k' k)
        ⊢ Eq (Option.map (fun a => Inhabited.default k') (GetElem?.getElem? L.reverse  …
      -/
      rw [ListBlank.nth_mk, List.getI_eq_iget_getElem?, List.map, List.reverse_nil]
      /-
        case neg
        K : Type u_1
        Γ : K → Type u_2
        Λ : Type u_3
        σ : Type u_4
        inst✝² : DecidableEq K
        inst✝¹ : Inhabited Λ
        inst✝ : Inhabited σ
        k : K
        L : List (Γ k)
        k' : K
        i : Nat
        this : Eq (Function.comp (Turing.proj k').f fun a => Function.update Inhabited …
        h : Not (Eq k' k)
        ⊢ Eq (Option.map (fun a => Inhabited.default k') (GetElem?.getElem? L.reverse  …
      -/
                              /-
                                🎉 no goals
                              -/
      cases L.reverse[i]? <;> rfl
                              /-
                                🎉 no goals
                              -/
    /-
      K : Type u_1
      Γ : K → Type u_2
      Λ : Type u_3
      σ : Type u_4
      inst✝² : DecidableEq K
      inst✝¹ : Inhabited Λ
      inst✝ : Inhabited σ
      k : K
      L : List (Γ k)
      ⊢ Eq (Turing.TM1.init (Turing.TM2to1.trInit k L)) { l := Option.map Turing.TM2 …
    -/
  · rw [trInit, TM1.init]
    /-
      K : Type u_1
      Γ : K → Type u_2
      Λ : Type u_3
      σ : Type u_4
      inst✝² : DecidableEq K
      inst✝¹ : Inhabited Λ
      inst✝ : Inhabited σ
      k : K
      L : List (Γ k)
      ⊢ Eq { l := Option.some Inhabited.default, var := Inhabited.default, Tape := T …
    -/
                                  /-
                                    🎉 no goals
                                  -/
                                  /-
                                    🎉 no goals
                                  -/
                                  /-
                                    🎉 no goals
                                  -/
    congr <;> cases L.reverse <;> try rfl
    /-
      case e_Tape.e_l.e_tail.cons
      K : Type u_1
      Γ : K → Type u_2
      Λ : Type u_3
      σ : Type u_4
      inst✝² : DecidableEq K
      inst✝¹ : Inhabited Λ
      inst✝ : Inhabited σ
      k : K
      L : List (Γ k)
      head✝ : Γ k
      tail✝ : List (Γ k)
      ⊢ Eq (List.map (fun a => { fst := Bool.false, snd := Function.update (fun x => …
    -/
    simp only [List.map_map, List.tail_cons, List.map]
    /-
      case e_Tape.e_l.e_tail.cons
      K : Type u_1
      Γ : K → Type u_2
      Λ : Type u_3
      σ : Type u_4
      inst✝² : DecidableEq K
      inst✝¹ : Inhabited Λ
      inst✝ : Inhabited σ
      k : K
      L : List (Γ k)
      head✝ : Γ k
      tail✝ : List (Γ k)
      ⊢ Eq (List.map (fun a => { fst := Bool.false, snd := Function.update (fun x => …
    -/
    rfl
    /-
      🎉 no goals
    -/


theorem tr_eval_dom (k) (L : List (Γ k)) :
    (TM1.eval (tr M) (trInit k L)).Dom ↔ (TM2.eval M k L).Dom :=
  Turing.tr_eval_dom (tr_respects M) (trCfg_init k L)


theorem tr_eval (k) (L : List (Γ k)) {L₁ L₂} (H₁ : L₁ ∈ TM1.eval (tr M) (trInit k L))
    (H₂ : L₂ ∈ TM2.eval M k L) :
    ∃ (S : ∀ k, List (Γ k)) (L' : ListBlank (∀ k, Option (Γ k))),
      addBottom L' = L₁ ∧
        (∀ k, L'.map (proj k) = ListBlank.mk ((S k).map some).reverse) ∧ S k = L₂ := by
  /-
    K : Type u_1
    Γ : K → Type u_2
    Λ : Type u_3
    σ : Type u_4
    inst✝² : DecidableEq K
    M : Λ → Turing.TM2.Stmt Γ Λ σ
    inst✝¹ : Inhabited Λ
    inst✝ : Inhabited σ
    k : K
    L : List (Γ k)
    L₁ : Turing.ListBlank Turing.TM2to1.Γ'
    L₂ : List (Γ k)
    H₁ : Membership.mem (Turing.TM1.eval (Turing.TM2to1.tr M) (Turing.TM2to1.trIni …
    H₂ : Membership.mem (Turing.TM2.eval M k L) L₂
    ⊢ Exists fun S => Exists fun L' => And (Eq (Turing.TM2to1.addBottom L') L₁) (A …
  -/
  obtain ⟨c₁, h₁, rfl⟩ := (Part.mem_map_iff _).1 H₁
  /-
    case intro.intro
    K : Type u_1
    Γ : K → Type u_2
    Λ : Type u_3
    σ : Type u_4
    inst✝² : DecidableEq K
    M : Λ → Turing.TM2.Stmt Γ Λ σ
    inst✝¹ : Inhabited Λ
    inst✝ : Inhabited σ
    k : K
    L L₂ : List (Γ k)
    H₂ : Membership.mem (Turing.TM2.eval M k L) L₂
    c₁ : Turing.TM1.Cfg Turing.TM2to1.Γ' Turing.TM2to1.Λ' σ
    h₁ : Membership.mem (Turing.eval (Turing.TM1.step (Turing.TM2to1.tr M)) (Turin …
    H₁ : Membership.mem (Turing.TM1.eval (Turing.TM2to1.tr M) (Turing.TM2to1.trIni …
    ⊢ Exists fun S => Exists fun L' => And (Eq (Turing.TM2to1.addBottom L') c₁.Tap …
  -/
  obtain ⟨c₂, h₂, rfl⟩ := (Part.mem_map_iff _).1 H₂
  /-
    case intro.intro.intro.intro
    K : Type u_1
    Γ : K → Type u_2
    Λ : Type u_3
    σ : Type u_4
    inst✝² : DecidableEq K
    M : Λ → Turing.TM2.Stmt Γ Λ σ
    inst✝¹ : Inhabited Λ
    inst✝ : Inhabited σ
    k : K
    L : List (Γ k)
    c₁ : Turing.TM1.Cfg Turing.TM2to1.Γ' Turing.TM2to1.Λ' σ
    h₁ : Membership.mem (Turing.eval (Turing.TM1.step (Turing.TM2to1.tr M)) (Turin …
    H₁ : Membership.mem (Turing.TM1.eval (Turing.TM2to1.tr M) (Turing.TM2to1.trIni …
    c₂ : Turing.TM2.Cfg Γ Λ σ
    h₂ : Membership.mem (Turing.eval (Turing.TM2.step M) (Turing.TM2.init k L)) c₂
    H₂ : Membership.mem (Turing.TM2.eval M k L) (c₂.stk k)
    ⊢ Exists fun S => Exists fun L' => And (Eq (Turing.TM2to1.addBottom L') c₁.Tap …
  -/
  obtain ⟨_, ⟨L', hT⟩, h₃⟩ := Turing.tr_eval (tr_respects M) (trCfg_init k L) h₂
  /-
    case intro.intro.intro.intro.intro.intro.mk
    K : Type u_1
    Γ : K → Type u_2
    Λ : Type u_3
    σ : Type u_4
    inst✝² : DecidableEq K
    M : Λ → Turing.TM2.Stmt Γ Λ σ
    inst✝¹ : Inhabited Λ
    inst✝ : Inhabited σ
    k : K
    L : List (Γ k)
    c₁ : Turing.TM1.Cfg Turing.TM2to1.Γ' Turing.TM2to1.Λ' σ
    h₁ : Membership.mem (Turing.eval (Turing.TM1.step (Turing.TM2to1.tr M)) (Turin …
    H₁ : Membership.mem (Turing.TM1.eval (Turing.TM2to1.tr M) (Turing.TM2to1.trIni …
    q✝ : Option Λ
    v✝ : σ
    S✝ : (k : K) → List (Γ k)
    L' : Turing.ListBlank ((k : K) → Option (Γ k))
    hT : ∀ (k : K), Eq (Turing.ListBlank.map (Turing.proj k) L') (Turing.ListBlank …
    h₂ : Membership.mem (Turing.eval (Turing.TM2.step M) (Turing.TM2.init k L)) {  …
    H₂ : Membership.mem (Turing.TM2.eval M k L) ({ l := q✝, var := v✝, stk := S✝ } …
    h₃ : Membership.mem (Turing.eval (Turing.TM1.step (Turing.TM2to1.tr M)) (Turin …
    ⊢ Exists fun S => Exists fun L' => And (Eq (Turing.TM2to1.addBottom L') c₁.Tap …
  -/
  cases Part.mem_unique h₁ h₃
  /-
    case intro.intro.intro.intro.intro.intro.mk.refl
    K : Type u_1
    Γ : K → Type u_2
    Λ : Type u_3
    σ : Type u_4
    inst✝² : DecidableEq K
    M : Λ → Turing.TM2.Stmt Γ Λ σ
    inst✝¹ : Inhabited Λ
    inst✝ : Inhabited σ
    k : K
    L : List (Γ k)
    q✝ : Option Λ
    v✝ : σ
    S✝ : (k : K) → List (Γ k)
    L' : Turing.ListBlank ((k : K) → Option (Γ k))
    hT : ∀ (k : K), Eq (Turing.ListBlank.map (Turing.proj k) L') (Turing.ListBlank …
    h₂ : Membership.mem (Turing.eval (Turing.TM2.step M) (Turing.TM2.init k L)) {  …
    H₂ : Membership.mem (Turing.TM2.eval M k L) ({ l := q✝, var := v✝, stk := S✝ } …
    h₃ : Membership.mem (Turing.eval (Turing.TM1.step (Turing.TM2to1.tr M)) (Turin …
    h₁ : Membership.mem (Turing.eval (Turing.TM1.step (Turing.TM2to1.tr M)) (Turin …
    H₁ : Membership.mem (Turing.TM1.eval (Turing.TM2to1.tr M) (Turing.TM2to1.trIni …
    ⊢ Exists fun S => Exists fun L'_1 => And (Eq (Turing.TM2to1.addBottom L'_1) {  …
  -/
  exact ⟨_, L', by simp only [Tape.mk'_right₀], hT, rfl⟩
  /-
    🎉 no goals
  -/


/-- The support of a set of TM2 states in the TM2 emulator. -/
noncomputable def trSupp (S : Finset Λ) : Finset Λ'₂₁ :=
  S.biUnion fun l ↦ insert (normal l) (trStmts₁ (M l))


theorem tr_supports {S} (ss : TM2.Supports M S) : TM1.Supports (tr M) (trSupp M S) :=
  ⟨Finset.mem_biUnion.2 ⟨_, ss.1, Finset.mem_insert.2 <| Or.inl rfl⟩, fun l' h ↦ by
    suffices ∀ (q) (_ : TM2.SupportsStmt S q) (_ : ∀ x ∈ trStmts₁ q, x ∈ trSupp M S),
        TM1.SupportsStmt (trSupp M S) (trNormal q) ∧
        ∀ l' ∈ trStmts₁ q, TM1.SupportsStmt (trSupp M S) (tr M l') by
      rcases Finset.mem_biUnion.1 h with ⟨l, lS, h⟩
      have :=
        this _ (ss.2 l lS) fun x hx ↦ Finset.mem_biUnion.2 ⟨_, lS, Finset.mem_insert_of_mem hx⟩
      rcases Finset.mem_insert.1 h with (rfl | h) <;> [exact this.1; exact this.2 _ h]
    /-
      K : Type u_1
      Γ : K → Type u_2
      Λ : Type u_3
      σ : Type u_4
      inst✝¹ : DecidableEq K
      M : Λ → Turing.TM2.Stmt Γ Λ σ
      inst✝ : Inhabited Λ
      S : Finset Λ
      ss : Turing.TM2.Supports M S
      l' : Turing.TM2to1.Λ'
      h : Membership.mem (Turing.TM2to1.trSupp M S) l'
      ⊢ ∀ (q : Turing.TM2.Stmt Γ Λ σ), Turing.TM2.SupportsStmt S q → (∀ (x : Turing. …
    -/
    clear h l'
    /-
      K : Type u_1
      Γ : K → Type u_2
      Λ : Type u_3
      σ : Type u_4
      inst✝¹ : DecidableEq K
      M : Λ → Turing.TM2.Stmt Γ Λ σ
      inst✝ : Inhabited Λ
      S : Finset Λ
      ss : Turing.TM2.Supports M S
      ⊢ ∀ (q : Turing.TM2.Stmt Γ Λ σ), Turing.TM2.SupportsStmt S q → (∀ (x : Turing. …
    -/
    refine stmtStRec ?_ ?_ ?_ ?_ ?_
      /-
        case refine_1
        K : Type u_1
        Γ : K → Type u_2
        Λ : Type u_3
        σ : Type u_4
        inst✝¹ : DecidableEq K
        M : Λ → Turing.TM2.Stmt Γ Λ σ
        inst✝ : Inhabited Λ
        S : Finset Λ
        ss : Turing.TM2.Supports M S
        ⊢ ∀ (k : K) (s : Turing.TM2to1.StAct k) (q : Turing.TM2.Stmt Γ Λ σ), (Turing.T …
      -/
    · intro _ s _ IH ss' sub -- stack op
      /-
        case refine_1
        K : Type u_1
        Γ : K → Type u_2
        Λ : Type u_3
        σ : Type u_4
        inst✝¹ : DecidableEq K
        M : Λ → Turing.TM2.Stmt Γ Λ σ
        inst✝ : Inhabited Λ
        S : Finset Λ
        ss : Turing.TM2.Supports M S
        k✝ : K
        s : Turing.TM2to1.StAct k✝
        q✝ : Turing.TM2.Stmt Γ Λ σ
        IH : Turing.TM2.SupportsStmt S q✝ → (∀ (x : Turing.TM2to1.Λ'), Membership.mem  …
        ss' : Turing.TM2.SupportsStmt S (Turing.TM2to1.stRun s q✝)
        sub : ∀ (x : Turing.TM2to1.Λ'), Membership.mem (Turing.TM2to1.trStmts₁ (Turing …
        ⊢ And (Turing.TM1.SupportsStmt (Turing.TM2to1.trSupp M S) (Turing.TM2to1.trNor …
      -/
      rw [TM2to1.supports_run] at ss'
      simp only [TM2to1.trStmts₁_run, Finset.mem_union, Finset.mem_insert, Finset.mem_singleton]
        at sub
      /-
        case refine_1
        K : Type u_1
        Γ : K → Type u_2
        Λ : Type u_3
        σ : Type u_4
        inst✝¹ : DecidableEq K
        M : Λ → Turing.TM2.Stmt Γ Λ σ
        inst✝ : Inhabited Λ
        S : Finset Λ
        ss : Turing.TM2.Supports M S
        k✝ : K
        s : Turing.TM2to1.StAct k✝
        q✝ : Turing.TM2.Stmt Γ Λ σ
        IH : Turing.TM2.SupportsStmt S q✝ → (∀ (x : Turing.TM2to1.Λ'), Membership.mem  …
        ss' : Turing.TM2.SupportsStmt S q✝
        sub : ∀ (x : Turing.TM2to1.Λ'), Or (Or (Eq x (Turing.TM2to1.Λ'.go k✝ s q✝)) (E …
        ⊢ And (Turing.TM1.SupportsStmt (Turing.TM2to1.trSupp M S) (Turing.TM2to1.trNor …
      -/
      have hgo := sub _ (Or.inl <| Or.inl rfl)
      /-
        case refine_1
        K : Type u_1
        Γ : K → Type u_2
        Λ : Type u_3
        σ : Type u_4
        inst✝¹ : DecidableEq K
        M : Λ → Turing.TM2.Stmt Γ Λ σ
        inst✝ : Inhabited Λ
        S : Finset Λ
        ss : Turing.TM2.Supports M S
        k✝ : K
        s : Turing.TM2to1.StAct k✝
        q✝ : Turing.TM2.Stmt Γ Λ σ
        IH : Turing.TM2.SupportsStmt S q✝ → (∀ (x : Turing.TM2to1.Λ'), Membership.mem  …
        ss' : Turing.TM2.SupportsStmt S q✝
        sub : ∀ (x : Turing.TM2to1.Λ'), Or (Or (Eq x (Turing.TM2to1.Λ'.go k✝ s q✝)) (E …
        hgo : Membership.mem (Turing.TM2to1.trSupp M S) (Turing.TM2to1.Λ'.go k✝ s q✝)
        ⊢ And (Turing.TM1.SupportsStmt (Turing.TM2to1.trSupp M S) (Turing.TM2to1.trNor …
      -/
      have hret := sub _ (Or.inl <| Or.inr rfl)
      /-
        case refine_1
        K : Type u_1
        Γ : K → Type u_2
        Λ : Type u_3
        σ : Type u_4
        inst✝¹ : DecidableEq K
        M : Λ → Turing.TM2.Stmt Γ Λ σ
        inst✝ : Inhabited Λ
        S : Finset Λ
        ss : Turing.TM2.Supports M S
        k✝ : K
        s : Turing.TM2to1.StAct k✝
        q✝ : Turing.TM2.Stmt Γ Λ σ
        IH : Turing.TM2.SupportsStmt S q✝ → (∀ (x : Turing.TM2to1.Λ'), Membership.mem  …
        ss' : Turing.TM2.SupportsStmt S q✝
        sub : ∀ (x : Turing.TM2to1.Λ'), Or (Or (Eq x (Turing.TM2to1.Λ'.go k✝ s q✝)) (E …
        hgo : Membership.mem (Turing.TM2to1.trSupp M S) (Turing.TM2to1.Λ'.go k✝ s q✝)
        hret : Membership.mem (Turing.TM2to1.trSupp M S) (Turing.TM2to1.Λ'.ret q✝)
        ⊢ And (Turing.TM1.SupportsStmt (Turing.TM2to1.trSupp M S) (Turing.TM2to1.trNor …
      -/
      cases' IH ss' fun x hx ↦ sub x <| Or.inr hx with IH₁ IH₂
      /-
        case refine_1.intro
        K : Type u_1
        Γ : K → Type u_2
        Λ : Type u_3
        σ : Type u_4
        inst✝¹ : DecidableEq K
        M : Λ → Turing.TM2.Stmt Γ Λ σ
        inst✝ : Inhabited Λ
        S : Finset Λ
        ss : Turing.TM2.Supports M S
        k✝ : K
        s : Turing.TM2to1.StAct k✝
        q✝ : Turing.TM2.Stmt Γ Λ σ
        IH : Turing.TM2.SupportsStmt S q✝ → (∀ (x : Turing.TM2to1.Λ'), Membership.mem  …
        ss' : Turing.TM2.SupportsStmt S q✝
        sub : ∀ (x : Turing.TM2to1.Λ'), Or (Or (Eq x (Turing.TM2to1.Λ'.go k✝ s q✝)) (E …
        hgo : Membership.mem (Turing.TM2to1.trSupp M S) (Turing.TM2to1.Λ'.go k✝ s q✝)
        hret : Membership.mem (Turing.TM2to1.trSupp M S) (Turing.TM2to1.Λ'.ret q✝)
        IH₁ : Turing.TM1.SupportsStmt (Turing.TM2to1.trSupp M S) (Turing.TM2to1.trNorm …
        IH₂ : ∀ (l' : Turing.TM2to1.Λ'), Membership.mem (Turing.TM2to1.trStmts₁ q✝) l' …
        ⊢ And (Turing.TM1.SupportsStmt (Turing.TM2to1.trSupp M S) (Turing.TM2to1.trNor …
      -/
      refine ⟨by simp only [trNormal_run, TM1.SupportsStmt]; intros; exact hgo, fun l h ↦ ?_⟩
      /-
        case refine_1.intro
        K : Type u_1
        Γ : K → Type u_2
        Λ : Type u_3
        σ : Type u_4
        inst✝¹ : DecidableEq K
        M : Λ → Turing.TM2.Stmt Γ Λ σ
        inst✝ : Inhabited Λ
        S : Finset Λ
        ss : Turing.TM2.Supports M S
        k✝ : K
        s : Turing.TM2to1.StAct k✝
        q✝ : Turing.TM2.Stmt Γ Λ σ
        IH : Turing.TM2.SupportsStmt S q✝ → (∀ (x : Turing.TM2to1.Λ'), Membership.mem  …
        ss' : Turing.TM2.SupportsStmt S q✝
        sub : ∀ (x : Turing.TM2to1.Λ'), Or (Or (Eq x (Turing.TM2to1.Λ'.go k✝ s q✝)) (E …
        hgo : Membership.mem (Turing.TM2to1.trSupp M S) (Turing.TM2to1.Λ'.go k✝ s q✝)
        hret : Membership.mem (Turing.TM2to1.trSupp M S) (Turing.TM2to1.Λ'.ret q✝)
        IH₁ : Turing.TM1.SupportsStmt (Turing.TM2to1.trSupp M S) (Turing.TM2to1.trNorm …
        IH₂ : ∀ (l' : Turing.TM2to1.Λ'), Membership.mem (Turing.TM2to1.trStmts₁ q✝) l' …
        l : Turing.TM2to1.Λ'
        h : Membership.mem (Turing.TM2to1.trStmts₁ (Turing.TM2to1.stRun s q✝)) l
        ⊢ Turing.TM1.SupportsStmt (Turing.TM2to1.trSupp M S) (Turing.TM2to1.tr M l)
      -/
      rw [trStmts₁_run] at h
      simp only [TM2to1.trStmts₁_run, Finset.mem_union, Finset.mem_insert, Finset.mem_singleton]
        at h
      /-
        case refine_1.intro
        K : Type u_1
        Γ : K → Type u_2
        Λ : Type u_3
        σ : Type u_4
        inst✝¹ : DecidableEq K
        M : Λ → Turing.TM2.Stmt Γ Λ σ
        inst✝ : Inhabited Λ
        S : Finset Λ
        ss : Turing.TM2.Supports M S
        k✝ : K
        s : Turing.TM2to1.StAct k✝
        q✝ : Turing.TM2.Stmt Γ Λ σ
        IH : Turing.TM2.SupportsStmt S q✝ → (∀ (x : Turing.TM2to1.Λ'), Membership.mem  …
        ss' : Turing.TM2.SupportsStmt S q✝
        sub : ∀ (x : Turing.TM2to1.Λ'), Or (Or (Eq x (Turing.TM2to1.Λ'.go k✝ s q✝)) (E …
        hgo : Membership.mem (Turing.TM2to1.trSupp M S) (Turing.TM2to1.Λ'.go k✝ s q✝)
        hret : Membership.mem (Turing.TM2to1.trSupp M S) (Turing.TM2to1.Λ'.ret q✝)
        IH₁ : Turing.TM1.SupportsStmt (Turing.TM2to1.trSupp M S) (Turing.TM2to1.trNorm …
        IH₂ : ∀ (l' : Turing.TM2to1.Λ'), Membership.mem (Turing.TM2to1.trStmts₁ q✝) l' …
        l : Turing.TM2to1.Λ'
        h : Or (Or (Eq l (Turing.TM2to1.Λ'.go k✝ s q✝)) (Eq l (Turing.TM2to1.Λ'.ret q✝ …
        ⊢ Turing.TM1.SupportsStmt (Turing.TM2to1.trSupp M S) (Turing.TM2to1.tr M l)
      -/
      rcases h with (⟨rfl | rfl⟩ | h)
        /-
          case refine_1.intro.inl.inl
          K : Type u_1
          Γ : K → Type u_2
          Λ : Type u_3
          σ : Type u_4
          inst✝¹ : DecidableEq K
          M : Λ → Turing.TM2.Stmt Γ Λ σ
          inst✝ : Inhabited Λ
          S : Finset Λ
          ss : Turing.TM2.Supports M S
          k✝ : K
          s : Turing.TM2to1.StAct k✝
          q✝ : Turing.TM2.Stmt Γ Λ σ
          IH : Turing.TM2.SupportsStmt S q✝ → (∀ (x : Turing.TM2to1.Λ'), Membership.mem  …
          ss' : Turing.TM2.SupportsStmt S q✝
          sub : ∀ (x : Turing.TM2to1.Λ'), Or (Or (Eq x (Turing.TM2to1.Λ'.go k✝ s q✝)) (E …
          hgo : Membership.mem (Turing.TM2to1.trSupp M S) (Turing.TM2to1.Λ'.go k✝ s q✝)
          hret : Membership.mem (Turing.TM2to1.trSupp M S) (Turing.TM2to1.Λ'.ret q✝)
          IH₁ : Turing.TM1.SupportsStmt (Turing.TM2to1.trSupp M S) (Turing.TM2to1.trNorm …
          IH₂ : ∀ (l' : Turing.TM2to1.Λ'), Membership.mem (Turing.TM2to1.trStmts₁ q✝) l' …
          ⊢ Turing.TM1.SupportsStmt (Turing.TM2to1.trSupp M S) (Turing.TM2to1.tr M (Turi …
        -/
      · cases s
          /-
            case refine_1.intro.inl.inl.push
            K : Type u_1
            Γ : K → Type u_2
            Λ : Type u_3
            σ : Type u_4
            inst✝¹ : DecidableEq K
            M : Λ → Turing.TM2.Stmt Γ Λ σ
            inst✝ : Inhabited Λ
            S : Finset Λ
            ss : Turing.TM2.Supports M S
            k✝ : K
            q✝ : Turing.TM2.Stmt Γ Λ σ
            IH : Turing.TM2.SupportsStmt S q✝ → (∀ (x : Turing.TM2to1.Λ'), Membership.mem  …
            ss' : Turing.TM2.SupportsStmt S q✝
            hret : Membership.mem (Turing.TM2to1.trSupp M S) (Turing.TM2to1.Λ'.ret q✝)
            IH₁ : Turing.TM1.SupportsStmt (Turing.TM2to1.trSupp M S) (Turing.TM2to1.trNorm …
            IH₂ : ∀ (l' : Turing.TM2to1.Λ'), Membership.mem (Turing.TM2to1.trStmts₁ q✝) l' …
            a✝ : σ → Γ k✝
            sub : ∀ (x : Turing.TM2to1.Λ'), Or (Or (Eq x (Turing.TM2to1.Λ'.go k✝ (Turing.T …
            hgo : Membership.mem (Turing.TM2to1.trSupp M S) (Turing.TM2to1.Λ'.go k✝ (Turin …
            ⊢ Turing.TM1.SupportsStmt (Turing.TM2to1.trSupp M S) (Turing.TM2to1.tr M (Turi …
          -/
        · exact ⟨fun _ _ ↦ hret, fun _ _ ↦ hgo⟩
          /-
            🎉 no goals
          -/
          /-
            case refine_1.intro.inl.inl.peek
            K : Type u_1
            Γ : K → Type u_2
            Λ : Type u_3
            σ : Type u_4
            inst✝¹ : DecidableEq K
            M : Λ → Turing.TM2.Stmt Γ Λ σ
            inst✝ : Inhabited Λ
            S : Finset Λ
            ss : Turing.TM2.Supports M S
            k✝ : K
            q✝ : Turing.TM2.Stmt Γ Λ σ
            IH : Turing.TM2.SupportsStmt S q✝ → (∀ (x : Turing.TM2to1.Λ'), Membership.mem  …
            ss' : Turing.TM2.SupportsStmt S q✝
            hret : Membership.mem (Turing.TM2to1.trSupp M S) (Turing.TM2to1.Λ'.ret q✝)
            IH₁ : Turing.TM1.SupportsStmt (Turing.TM2to1.trSupp M S) (Turing.TM2to1.trNorm …
            IH₂ : ∀ (l' : Turing.TM2to1.Λ'), Membership.mem (Turing.TM2to1.trStmts₁ q✝) l' …
            a✝ : σ → Option (Γ k✝) → σ
            sub : ∀ (x : Turing.TM2to1.Λ'), Or (Or (Eq x (Turing.TM2to1.Λ'.go k✝ (Turing.T …
            hgo : Membership.mem (Turing.TM2to1.trSupp M S) (Turing.TM2to1.Λ'.go k✝ (Turin …
            ⊢ Turing.TM1.SupportsStmt (Turing.TM2to1.trSupp M S) (Turing.TM2to1.tr M (Turi …
          -/
        · exact ⟨fun _ _ ↦ hret, fun _ _ ↦ hgo⟩
          /-
            🎉 no goals
          -/
          /-
            case refine_1.intro.inl.inl.pop
            K : Type u_1
            Γ : K → Type u_2
            Λ : Type u_3
            σ : Type u_4
            inst✝¹ : DecidableEq K
            M : Λ → Turing.TM2.Stmt Γ Λ σ
            inst✝ : Inhabited Λ
            S : Finset Λ
            ss : Turing.TM2.Supports M S
            k✝ : K
            q✝ : Turing.TM2.Stmt Γ Λ σ
            IH : Turing.TM2.SupportsStmt S q✝ → (∀ (x : Turing.TM2to1.Λ'), Membership.mem  …
            ss' : Turing.TM2.SupportsStmt S q✝
            hret : Membership.mem (Turing.TM2to1.trSupp M S) (Turing.TM2to1.Λ'.ret q✝)
            IH₁ : Turing.TM1.SupportsStmt (Turing.TM2to1.trSupp M S) (Turing.TM2to1.trNorm …
            IH₂ : ∀ (l' : Turing.TM2to1.Λ'), Membership.mem (Turing.TM2to1.trStmts₁ q✝) l' …
            a✝ : σ → Option (Γ k✝) → σ
            sub : ∀ (x : Turing.TM2to1.Λ'), Or (Or (Eq x (Turing.TM2to1.Λ'.go k✝ (Turing.T …
            hgo : Membership.mem (Turing.TM2to1.trSupp M S) (Turing.TM2to1.Λ'.go k✝ (Turin …
            ⊢ Turing.TM1.SupportsStmt (Turing.TM2to1.trSupp M S) (Turing.TM2to1.tr M (Turi …
          -/
        · exact ⟨⟨fun _ _ ↦ hret, fun _ _ ↦ hret⟩, fun _ _ ↦ hgo⟩
          /-
            🎉 no goals
          -/
        /-
          case refine_1.intro.inl.inr
          K : Type u_1
          Γ : K → Type u_2
          Λ : Type u_3
          σ : Type u_4
          inst✝¹ : DecidableEq K
          M : Λ → Turing.TM2.Stmt Γ Λ σ
          inst✝ : Inhabited Λ
          S : Finset Λ
          ss : Turing.TM2.Supports M S
          k✝ : K
          s : Turing.TM2to1.StAct k✝
          q✝ : Turing.TM2.Stmt Γ Λ σ
          IH : Turing.TM2.SupportsStmt S q✝ → (∀ (x : Turing.TM2to1.Λ'), Membership.mem  …
          ss' : Turing.TM2.SupportsStmt S q✝
          sub : ∀ (x : Turing.TM2to1.Λ'), Or (Or (Eq x (Turing.TM2to1.Λ'.go k✝ s q✝)) (E …
          hgo : Membership.mem (Turing.TM2to1.trSupp M S) (Turing.TM2to1.Λ'.go k✝ s q✝)
          hret : Membership.mem (Turing.TM2to1.trSupp M S) (Turing.TM2to1.Λ'.ret q✝)
          IH₁ : Turing.TM1.SupportsStmt (Turing.TM2to1.trSupp M S) (Turing.TM2to1.trNorm …
          IH₂ : ∀ (l' : Turing.TM2to1.Λ'), Membership.mem (Turing.TM2to1.trStmts₁ q✝) l' …
          ⊢ Turing.TM1.SupportsStmt (Turing.TM2to1.trSupp M S) (Turing.TM2to1.tr M (Turi …
        -/
      · unfold TM1.SupportsStmt TM2to1.tr
        /-
          case refine_1.intro.inl.inr
          K : Type u_1
          Γ : K → Type u_2
          Λ : Type u_3
          σ : Type u_4
          inst✝¹ : DecidableEq K
          M : Λ → Turing.TM2.Stmt Γ Λ σ
          inst✝ : Inhabited Λ
          S : Finset Λ
          ss : Turing.TM2.Supports M S
          k✝ : K
          s : Turing.TM2to1.StAct k✝
          q✝ : Turing.TM2.Stmt Γ Λ σ
          IH : Turing.TM2.SupportsStmt S q✝ → (∀ (x : Turing.TM2to1.Λ'), Membership.mem  …
          ss' : Turing.TM2.SupportsStmt S q✝
          sub : ∀ (x : Turing.TM2to1.Λ'), Or (Or (Eq x (Turing.TM2to1.Λ'.go k✝ s q✝)) (E …
          hgo : Membership.mem (Turing.TM2to1.trSupp M S) (Turing.TM2to1.Λ'.go k✝ s q✝)
          hret : Membership.mem (Turing.TM2to1.trSupp M S) (Turing.TM2to1.Λ'.ret q✝)
          IH₁ : Turing.TM1.SupportsStmt (Turing.TM2to1.trSupp M S) (Turing.TM2to1.trNorm …
          IH₂ : ∀ (l' : Turing.TM2to1.Λ'), Membership.mem (Turing.TM2to1.trStmts₁ q✝) l' …
          ⊢ Turing.TM1.SupportsStmt.match_1 (fun x => Prop) (Turing.TM2to1.tr.match_1 (f …
        -/
        exact ⟨IH₁, fun _ _ ↦ hret⟩
        /-
          🎉 no goals
        -/
        /-
          case refine_1.intro.inr
          K : Type u_1
          Γ : K → Type u_2
          Λ : Type u_3
          σ : Type u_4
          inst✝¹ : DecidableEq K
          M : Λ → Turing.TM2.Stmt Γ Λ σ
          inst✝ : Inhabited Λ
          S : Finset Λ
          ss : Turing.TM2.Supports M S
          k✝ : K
          s : Turing.TM2to1.StAct k✝
          q✝ : Turing.TM2.Stmt Γ Λ σ
          IH : Turing.TM2.SupportsStmt S q✝ → (∀ (x : Turing.TM2to1.Λ'), Membership.mem  …
          ss' : Turing.TM2.SupportsStmt S q✝
          sub : ∀ (x : Turing.TM2to1.Λ'), Or (Or (Eq x (Turing.TM2to1.Λ'.go k✝ s q✝)) (E …
          hgo : Membership.mem (Turing.TM2to1.trSupp M S) (Turing.TM2to1.Λ'.go k✝ s q✝)
          hret : Membership.mem (Turing.TM2to1.trSupp M S) (Turing.TM2to1.Λ'.ret q✝)
          IH₁ : Turing.TM1.SupportsStmt (Turing.TM2to1.trSupp M S) (Turing.TM2to1.trNorm …
          IH₂ : ∀ (l' : Turing.TM2to1.Λ'), Membership.mem (Turing.TM2to1.trStmts₁ q✝) l' …
          l : Turing.TM2to1.Λ'
          h : Membership.mem (Turing.TM2to1.trStmts₁ q✝) l
          ⊢ Turing.TM1.SupportsStmt (Turing.TM2to1.trSupp M S) (Turing.TM2to1.tr M l)
        -/
      · exact IH₂ _ h
        /-
          🎉 no goals
        -/
      /-
        case refine_2
        K : Type u_1
        Γ : K → Type u_2
        Λ : Type u_3
        σ : Type u_4
        inst✝¹ : DecidableEq K
        M : Λ → Turing.TM2.Stmt Γ Λ σ
        inst✝ : Inhabited Λ
        S : Finset Λ
        ss : Turing.TM2.Supports M S
        ⊢ ∀ (a : σ → σ) (q : Turing.TM2.Stmt Γ Λ σ), (Turing.TM2.SupportsStmt S q → (∀ …
      -/
    · intro _ _ IH ss' sub -- load
      /-
        case refine_2
        K : Type u_1
        Γ : K → Type u_2
        Λ : Type u_3
        σ : Type u_4
        inst✝¹ : DecidableEq K
        M : Λ → Turing.TM2.Stmt Γ Λ σ
        inst✝ : Inhabited Λ
        S : Finset Λ
        ss : Turing.TM2.Supports M S
        a✝ : σ → σ
        q✝ : Turing.TM2.Stmt Γ Λ σ
        IH : Turing.TM2.SupportsStmt S q✝ → (∀ (x : Turing.TM2to1.Λ'), Membership.mem  …
        ss' : Turing.TM2.SupportsStmt S (Turing.TM2.Stmt.load a✝ q✝)
        sub : ∀ (x : Turing.TM2to1.Λ'), Membership.mem (Turing.TM2to1.trStmts₁ (Turing …
        ⊢ And (Turing.TM1.SupportsStmt (Turing.TM2to1.trSupp M S) (Turing.TM2to1.trNor …
      -/
      unfold TM2to1.trStmts₁ at sub ⊢
      /-
        case refine_2
        K : Type u_1
        Γ : K → Type u_2
        Λ : Type u_3
        σ : Type u_4
        inst✝¹ : DecidableEq K
        M : Λ → Turing.TM2.Stmt Γ Λ σ
        inst✝ : Inhabited Λ
        S : Finset Λ
        ss : Turing.TM2.Supports M S
        a✝ : σ → σ
        q✝ : Turing.TM2.Stmt Γ Λ σ
        IH : Turing.TM2.SupportsStmt S q✝ → (∀ (x : Turing.TM2to1.Λ'), Membership.mem  …
        ss' : Turing.TM2.SupportsStmt S (Turing.TM2.Stmt.load a✝ q✝)
        sub : ∀ (x : Turing.TM2to1.Λ'), Membership.mem (Turing.TM2to1.trStmts₁ q✝) x → …
        ⊢ And (Turing.TM1.SupportsStmt (Turing.TM2to1.trSupp M S) (Turing.TM2to1.trNor …
      -/
      exact IH ss' sub
      /-
        🎉 no goals
      -/
      /-
        case refine_3
        K : Type u_1
        Γ : K → Type u_2
        Λ : Type u_3
        σ : Type u_4
        inst✝¹ : DecidableEq K
        M : Λ → Turing.TM2.Stmt Γ Λ σ
        inst✝ : Inhabited Λ
        S : Finset Λ
        ss : Turing.TM2.Supports M S
        ⊢ ∀ (p : σ → Bool) (q₁ q₂ : Turing.TM2.Stmt Γ Λ σ), (Turing.TM2.SupportsStmt S …
      -/
    · intro _ _ _ IH₁ IH₂ ss' sub -- branch
      /-
        case refine_3
        K : Type u_1
        Γ : K → Type u_2
        Λ : Type u_3
        σ : Type u_4
        inst✝¹ : DecidableEq K
        M : Λ → Turing.TM2.Stmt Γ Λ σ
        inst✝ : Inhabited Λ
        S : Finset Λ
        ss : Turing.TM2.Supports M S
        p✝ : σ → Bool
        q₁✝ q₂✝ : Turing.TM2.Stmt Γ Λ σ
        IH₁ : Turing.TM2.SupportsStmt S q₁✝ → (∀ (x : Turing.TM2to1.Λ'), Membership.me …
        IH₂ : Turing.TM2.SupportsStmt S q₂✝ → (∀ (x : Turing.TM2to1.Λ'), Membership.me …
        ss' : Turing.TM2.SupportsStmt S (Turing.TM2.Stmt.branch p✝ q₁✝ q₂✝)
        sub : ∀ (x : Turing.TM2to1.Λ'), Membership.mem (Turing.TM2to1.trStmts₁ (Turing …
        ⊢ And (Turing.TM1.SupportsStmt (Turing.TM2to1.trSupp M S) (Turing.TM2to1.trNor …
      -/
      unfold TM2to1.trStmts₁ at sub
      /-
        case refine_3
        K : Type u_1
        Γ : K → Type u_2
        Λ : Type u_3
        σ : Type u_4
        inst✝¹ : DecidableEq K
        M : Λ → Turing.TM2.Stmt Γ Λ σ
        inst✝ : Inhabited Λ
        S : Finset Λ
        ss : Turing.TM2.Supports M S
        p✝ : σ → Bool
        q₁✝ q₂✝ : Turing.TM2.Stmt Γ Λ σ
        IH₁ : Turing.TM2.SupportsStmt S q₁✝ → (∀ (x : Turing.TM2to1.Λ'), Membership.me …
        IH₂ : Turing.TM2.SupportsStmt S q₂✝ → (∀ (x : Turing.TM2to1.Λ'), Membership.me …
        ss' : Turing.TM2.SupportsStmt S (Turing.TM2.Stmt.branch p✝ q₁✝ q₂✝)
        sub : ∀ (x : Turing.TM2to1.Λ'), Membership.mem (Union.union (Turing.TM2to1.trS …
        ⊢ And (Turing.TM1.SupportsStmt (Turing.TM2to1.trSupp M S) (Turing.TM2to1.trNor …
      -/
      cases' IH₁ ss'.1 fun x hx ↦ sub x <| Finset.mem_union_left _ hx with IH₁₁ IH₁₂
      /-
        case refine_3.intro
        K : Type u_1
        Γ : K → Type u_2
        Λ : Type u_3
        σ : Type u_4
        inst✝¹ : DecidableEq K
        M : Λ → Turing.TM2.Stmt Γ Λ σ
        inst✝ : Inhabited Λ
        S : Finset Λ
        ss : Turing.TM2.Supports M S
        p✝ : σ → Bool
        q₁✝ q₂✝ : Turing.TM2.Stmt Γ Λ σ
        IH₁ : Turing.TM2.SupportsStmt S q₁✝ → (∀ (x : Turing.TM2to1.Λ'), Membership.me …
        IH₂ : Turing.TM2.SupportsStmt S q₂✝ → (∀ (x : Turing.TM2to1.Λ'), Membership.me …
        ss' : Turing.TM2.SupportsStmt S (Turing.TM2.Stmt.branch p✝ q₁✝ q₂✝)
        sub : ∀ (x : Turing.TM2to1.Λ'), Membership.mem (Union.union (Turing.TM2to1.trS …
        IH₁₁ : Turing.TM1.SupportsStmt (Turing.TM2to1.trSupp M S) (Turing.TM2to1.trNor …
        IH₁₂ : ∀ (l' : Turing.TM2to1.Λ'), Membership.mem (Turing.TM2to1.trStmts₁ q₁✝)  …
        ⊢ And (Turing.TM1.SupportsStmt (Turing.TM2to1.trSupp M S) (Turing.TM2to1.trNor …
      -/
      cases' IH₂ ss'.2 fun x hx ↦ sub x <| Finset.mem_union_right _ hx with IH₂₁ IH₂₂
      /-
        case refine_3.intro.intro
        K : Type u_1
        Γ : K → Type u_2
        Λ : Type u_3
        σ : Type u_4
        inst✝¹ : DecidableEq K
        M : Λ → Turing.TM2.Stmt Γ Λ σ
        inst✝ : Inhabited Λ
        S : Finset Λ
        ss : Turing.TM2.Supports M S
        p✝ : σ → Bool
        q₁✝ q₂✝ : Turing.TM2.Stmt Γ Λ σ
        IH₁ : Turing.TM2.SupportsStmt S q₁✝ → (∀ (x : Turing.TM2to1.Λ'), Membership.me …
        IH₂ : Turing.TM2.SupportsStmt S q₂✝ → (∀ (x : Turing.TM2to1.Λ'), Membership.me …
        ss' : Turing.TM2.SupportsStmt S (Turing.TM2.Stmt.branch p✝ q₁✝ q₂✝)
        sub : ∀ (x : Turing.TM2to1.Λ'), Membership.mem (Union.union (Turing.TM2to1.trS …
        IH₁₁ : Turing.TM1.SupportsStmt (Turing.TM2to1.trSupp M S) (Turing.TM2to1.trNor …
        IH₁₂ : ∀ (l' : Turing.TM2to1.Λ'), Membership.mem (Turing.TM2to1.trStmts₁ q₁✝)  …
        IH₂₁ : Turing.TM1.SupportsStmt (Turing.TM2to1.trSupp M S) (Turing.TM2to1.trNor …
        IH₂₂ : ∀ (l' : Turing.TM2to1.Λ'), Membership.mem (Turing.TM2to1.trStmts₁ q₂✝)  …
        ⊢ And (Turing.TM1.SupportsStmt (Turing.TM2to1.trSupp M S) (Turing.TM2to1.trNor …
      -/
      refine ⟨⟨IH₁₁, IH₂₁⟩, fun l h ↦ ?_⟩
      /-
        case refine_3.intro.intro
        K : Type u_1
        Γ : K → Type u_2
        Λ : Type u_3
        σ : Type u_4
        inst✝¹ : DecidableEq K
        M : Λ → Turing.TM2.Stmt Γ Λ σ
        inst✝ : Inhabited Λ
        S : Finset Λ
        ss : Turing.TM2.Supports M S
        p✝ : σ → Bool
        q₁✝ q₂✝ : Turing.TM2.Stmt Γ Λ σ
        IH₁ : Turing.TM2.SupportsStmt S q₁✝ → (∀ (x : Turing.TM2to1.Λ'), Membership.me …
        IH₂ : Turing.TM2.SupportsStmt S q₂✝ → (∀ (x : Turing.TM2to1.Λ'), Membership.me …
        ss' : Turing.TM2.SupportsStmt S (Turing.TM2.Stmt.branch p✝ q₁✝ q₂✝)
        sub : ∀ (x : Turing.TM2to1.Λ'), Membership.mem (Union.union (Turing.TM2to1.trS …
        IH₁₁ : Turing.TM1.SupportsStmt (Turing.TM2to1.trSupp M S) (Turing.TM2to1.trNor …
        IH₁₂ : ∀ (l' : Turing.TM2to1.Λ'), Membership.mem (Turing.TM2to1.trStmts₁ q₁✝)  …
        IH₂₁ : Turing.TM1.SupportsStmt (Turing.TM2to1.trSupp M S) (Turing.TM2to1.trNor …
        IH₂₂ : ∀ (l' : Turing.TM2to1.Λ'), Membership.mem (Turing.TM2to1.trStmts₁ q₂✝)  …
        l : Turing.TM2to1.Λ'
        h : Membership.mem (Turing.TM2to1.trStmts₁ (Turing.TM2.Stmt.branch p✝ q₁✝ q₂✝) …
        ⊢ Turing.TM1.SupportsStmt (Turing.TM2to1.trSupp M S) (Turing.TM2to1.tr M l)
      -/
      rw [trStmts₁] at h
      /-
        case refine_3.intro.intro
        K : Type u_1
        Γ : K → Type u_2
        Λ : Type u_3
        σ : Type u_4
        inst✝¹ : DecidableEq K
        M : Λ → Turing.TM2.Stmt Γ Λ σ
        inst✝ : Inhabited Λ
        S : Finset Λ
        ss : Turing.TM2.Supports M S
        p✝ : σ → Bool
        q₁✝ q₂✝ : Turing.TM2.Stmt Γ Λ σ
        IH₁ : Turing.TM2.SupportsStmt S q₁✝ → (∀ (x : Turing.TM2to1.Λ'), Membership.me …
        IH₂ : Turing.TM2.SupportsStmt S q₂✝ → (∀ (x : Turing.TM2to1.Λ'), Membership.me …
        ss' : Turing.TM2.SupportsStmt S (Turing.TM2.Stmt.branch p✝ q₁✝ q₂✝)
        sub : ∀ (x : Turing.TM2to1.Λ'), Membership.mem (Union.union (Turing.TM2to1.trS …
        IH₁₁ : Turing.TM1.SupportsStmt (Turing.TM2to1.trSupp M S) (Turing.TM2to1.trNor …
        IH₁₂ : ∀ (l' : Turing.TM2to1.Λ'), Membership.mem (Turing.TM2to1.trStmts₁ q₁✝)  …
        IH₂₁ : Turing.TM1.SupportsStmt (Turing.TM2to1.trSupp M S) (Turing.TM2to1.trNor …
        IH₂₂ : ∀ (l' : Turing.TM2to1.Λ'), Membership.mem (Turing.TM2to1.trStmts₁ q₂✝)  …
        l : Turing.TM2to1.Λ'
        h : Membership.mem (Union.union (Turing.TM2to1.trStmts₁ q₁✝) (Turing.TM2to1.tr …
        ⊢ Turing.TM1.SupportsStmt (Turing.TM2to1.trSupp M S) (Turing.TM2to1.tr M l)
      -/
      rcases Finset.mem_union.1 h with (h | h) <;> [exact IH₁₂ _ h; exact IH₂₂ _ h]
      /-
        🎉 no goals
      -/
      /-
        case refine_4
        K : Type u_1
        Γ : K → Type u_2
        Λ : Type u_3
        σ : Type u_4
        inst✝¹ : DecidableEq K
        M : Λ → Turing.TM2.Stmt Γ Λ σ
        inst✝ : Inhabited Λ
        S : Finset Λ
        ss : Turing.TM2.Supports M S
        ⊢ ∀ (l : σ → Λ), Turing.TM2.SupportsStmt S (Turing.TM2.Stmt.goto l) → (∀ (x :  …
      -/
    · intro _ ss' _ -- goto
      /-
        case refine_4
        K : Type u_1
        Γ : K → Type u_2
        Λ : Type u_3
        σ : Type u_4
        inst✝¹ : DecidableEq K
        M : Λ → Turing.TM2.Stmt Γ Λ σ
        inst✝ : Inhabited Λ
        S : Finset Λ
        ss : Turing.TM2.Supports M S
        l✝ : σ → Λ
        ss' : Turing.TM2.SupportsStmt S (Turing.TM2.Stmt.goto l✝)
        x✝ : ∀ (x : Turing.TM2to1.Λ'), Membership.mem (Turing.TM2to1.trStmts₁ (Turing. …
        ⊢ And (Turing.TM1.SupportsStmt (Turing.TM2to1.trSupp M S) (Turing.TM2to1.trNor …
      -/
      simp only [trStmts₁, Finset.not_mem_empty]; refine ⟨?_, fun _ ↦ False.elim⟩
      /-
        case refine_4
        K : Type u_1
        Γ : K → Type u_2
        Λ : Type u_3
        σ : Type u_4
        inst✝¹ : DecidableEq K
        M : Λ → Turing.TM2.Stmt Γ Λ σ
        inst✝ : Inhabited Λ
        S : Finset Λ
        ss : Turing.TM2.Supports M S
        l✝ : σ → Λ
        ss' : Turing.TM2.SupportsStmt S (Turing.TM2.Stmt.goto l✝)
        x✝ : ∀ (x : Turing.TM2to1.Λ'), Membership.mem (Turing.TM2to1.trStmts₁ (Turing. …
        ⊢ Turing.TM1.SupportsStmt (Turing.TM2to1.trSupp M S) (Turing.TM2to1.trNormal ( …
      -/
      exact fun _ v ↦ Finset.mem_biUnion.2 ⟨_, ss' v, Finset.mem_insert_self _ _⟩
      /-
        🎉 no goals
      -/
      /-
        case refine_5
        K : Type u_1
        Γ : K → Type u_2
        Λ : Type u_3
        σ : Type u_4
        inst✝¹ : DecidableEq K
        M : Λ → Turing.TM2.Stmt Γ Λ σ
        inst✝ : Inhabited Λ
        S : Finset Λ
        ss : Turing.TM2.Supports M S
        ⊢ Turing.TM2.SupportsStmt S Turing.TM2.Stmt.halt → (∀ (x : Turing.TM2to1.Λ'),  …
      -/
    · intro _ _ -- halt
      /-
        case refine_5
        K : Type u_1
        Γ : K → Type u_2
        Λ : Type u_3
        σ : Type u_4
        inst✝¹ : DecidableEq K
        M : Λ → Turing.TM2.Stmt Γ Λ σ
        inst✝ : Inhabited Λ
        S : Finset Λ
        ss : Turing.TM2.Supports M S
        x✝¹ : Turing.TM2.SupportsStmt S Turing.TM2.Stmt.halt
        x✝ : ∀ (x : Turing.TM2to1.Λ'), Membership.mem (Turing.TM2to1.trStmts₁ Turing.T …
        ⊢ And (Turing.TM1.SupportsStmt (Turing.TM2to1.trSupp M S) (Turing.TM2to1.trNor …
      -/
      simp only [trStmts₁, Finset.not_mem_empty]
      /-
        case refine_5
        K : Type u_1
        Γ : K → Type u_2
        Λ : Type u_3
        σ : Type u_4
        inst✝¹ : DecidableEq K
        M : Λ → Turing.TM2.Stmt Γ Λ σ
        inst✝ : Inhabited Λ
        S : Finset Λ
        ss : Turing.TM2.Supports M S
        x✝¹ : Turing.TM2.SupportsStmt S Turing.TM2.Stmt.halt
        x✝ : ∀ (x : Turing.TM2to1.Λ'), Membership.mem (Turing.TM2to1.trStmts₁ Turing.T …
        ⊢ And (Turing.TM1.SupportsStmt (Turing.TM2to1.trSupp M S) (Turing.TM2to1.trNor …
      -/
      exact ⟨trivial, fun _ ↦ False.elim⟩⟩
      /-
        🎉 no goals
      -/


