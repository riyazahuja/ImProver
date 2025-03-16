/-- Reduction step for the additive free group relation: `w + x + (-x) + v ~> w + v` -/
inductive FreeAddGroup.Red.Step : List (α × Bool) → List (α × Bool) → Prop
  | not {L₁ L₂ x b} : FreeAddGroup.Red.Step (L₁ ++ (x, b) :: (x, not b) :: L₂) (L₁ ++ L₂)


attribute [simp] FreeAddGroup.Red.Step.not


/-- Reduction step for the multiplicative free group relation: `w * x * x⁻¹ * v ~> w * v` -/
@[to_additive FreeAddGroup.Red.Step]
inductive FreeGroup.Red.Step : List (α × Bool) → List (α × Bool) → Prop
  | not {L₁ L₂ x b} : FreeGroup.Red.Step (L₁ ++ (x, b) :: (x, not b) :: L₂) (L₁ ++ L₂)


attribute [simp] FreeGroup.Red.Step.not


/-- Reflexive-transitive closure of `Red.Step` -/
@[to_additive FreeAddGroup.Red "Reflexive-transitive closure of `Red.Step`"]
def Red : List (α × Bool) → List (α × Bool) → Prop :=
  ReflTransGen Red.Step


@[to_additive (attr := refl)]
theorem Red.refl : Red L L :=
  ReflTransGen.refl


@[to_additive (attr := trans)]
theorem Red.trans : Red L₁ L₂ → Red L₂ L₃ → Red L₁ L₃ :=
  ReflTransGen.trans


/-- Predicate asserting that the word `w₁` can be reduced to `w₂` in one step, i.e. there are words
`w₃ w₄` and letter `x` such that `w₁ = w₃xx⁻¹w₄` and `w₂ = w₃w₄`  -/
@[to_additive "Predicate asserting that the word `w₁` can be reduced to `w₂` in one step, i.e. there
  are words `w₃ w₄` and letter `x` such that `w₁ = w₃ + x + (-x) + w₄` and `w₂ = w₃w₄`"]
theorem Step.length : ∀ {L₁ L₂ : List (α × Bool)}, Step L₁ L₂ → L₂.length + 2 = L₁.length
                                          /-
                                            α : Type u
                                            L1 L2 : List (Prod α Bool)
                                            x : α
                                            b : Bool
                                            ⊢ Eq (HAdd.hAdd (HAppend.hAppend L1 L2).length 2) (HAppend.hAppend L1 (List.co …
                                          -/
  | _, _, @Red.Step.not _ L1 L2 x b => by rw [List.length_append, List.length_append]; rfl
                                                                                       /-
                                                                                         🎉 no goals
                                                                                       -/


@[to_additive (attr := simp)]
theorem Step.not_rev {x b} : Step (L₁ ++ (x, !b) :: (x, b) :: L₂) (L₁ ++ L₂) := by
  /-
    α : Type u
    L₁ L₂ : List (Prod α Bool)
    x : α
    b : Bool
    ⊢ FreeGroup.Red.Step (HAppend.hAppend L₁ (List.cons { fst := x, snd := b.not } …
  -/
              /-
                🎉 no goals
              -/
  cases b <;> exact Step.not
              /-
                🎉 no goals
              -/


@[to_additive (attr := simp)]
theorem Step.cons_not {x b} : Red.Step ((x, b) :: (x, !b) :: L) L :=
  @Step.not _ [] _ _ _


@[to_additive (attr := simp)]
theorem Step.cons_not_rev {x b} : Red.Step ((x, !b) :: (x, b) :: L) L :=
  @Red.Step.not_rev _ [] _ _ _


@[to_additive]
theorem Step.append_left : ∀ {L₁ L₂ L₃ : List (α × Bool)}, Step L₂ L₃ → Step (L₁ ++ L₂) (L₁ ++ L₃)
                                /-
                                  α : Type u
                                  x✝¹ L₁✝ L₂✝ : List (Prod α Bool)
                                  x✝ : α
                                  b✝ : Bool
                                  ⊢ FreeGroup.Red.Step (HAppend.hAppend x✝¹ (HAppend.hAppend L₁✝ (List.cons { fs …
                                -/
  | _, _, _, Red.Step.not => by rw [← List.append_assoc, ← List.append_assoc]; constructor
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/


@[to_additive]
theorem Step.cons {x} (H : Red.Step L₁ L₂) : Red.Step (x :: L₁) (x :: L₂) :=
  @Step.append_left _ [x] _ _ H


@[to_additive]
theorem Step.append_right : ∀ {L₁ L₂ L₃ : List (α × Bool)}, Step L₁ L₂ → Step (L₁ ++ L₃) (L₂ ++ L₃)
                                /-
                                  α : Type u
                                  x✝¹ L₁✝ L₂✝ : List (Prod α Bool)
                                  x✝ : α
                                  b✝ : Bool
                                  ⊢ FreeGroup.Red.Step (HAppend.hAppend (HAppend.hAppend L₁✝ (List.cons { fst := …
                                -/
  | _, _, _, Red.Step.not => by simp
                                /-
                                  🎉 no goals
                                -/


@[to_additive]
theorem not_step_nil : ¬Step [] L := by
  /-
    α : Type u
    L : List (Prod α Bool)
    ⊢ Not (FreeGroup.Red.Step List.nil L)
  -/
  generalize h' : [] = L'
  /-
    α : Type u
    L L' : List (Prod α Bool)
    h' : Eq List.nil L'
    ⊢ Not (FreeGroup.Red.Step L' L)
  -/
  intro h
  /-
    α : Type u
    L L' : List (Prod α Bool)
    h' : Eq List.nil L'
    h : FreeGroup.Red.Step L' L
    ⊢ False
  -/
  cases' h with L₁ L₂
  /-
    case not
    α : Type u
    L₁ L₂ : List (Prod α Bool)
    x✝ : α
    b✝ : Bool
    h' : Eq List.nil (HAppend.hAppend L₁ (List.cons { fst := x✝, snd := b✝ } (List …
    ⊢ False
  -/
  simp [List.nil_eq_append_iff] at h'
  /-
    🎉 no goals
  -/


@[to_additive]
theorem Step.cons_left_iff {a : α} {b : Bool} :
    Step ((a, b) :: L₁) L₂ ↔ (∃ L, Step L₁ L ∧ L₂ = (a, b) :: L) ∨ L₁ = (a, ! b) :: L₂ := by
  /-
    α : Type u
    L₁ L₂ : List (Prod α Bool)
    a : α
    b : Bool
    ⊢ Iff (FreeGroup.Red.Step (List.cons { fst := a, snd := b } L₁) L₂) (Or (Exist …
  -/
  constructor
    /-
      case mp
      α : Type u
      L₁ L₂ : List (Prod α Bool)
      a : α
      b : Bool
      ⊢ FreeGroup.Red.Step (List.cons { fst := a, snd := b } L₁) L₂ → Or (Exists fun …
    -/
  · generalize hL : ((a, b) :: L₁ : List _) = L
    /-
      case mp
      α : Type u
      L₁ L₂ : List (Prod α Bool)
      a : α
      b : Bool
      L : List (Prod α Bool)
      hL : Eq (List.cons { fst := a, snd := b } L₁) L
      ⊢ FreeGroup.Red.Step L L₂ → Or (Exists fun L => And (FreeGroup.Red.Step L₁ L)  …
    -/
    rintro @⟨_ | ⟨p, s'⟩, e, a', b'⟩
      /-
        case mp.not.nil
        α : Type u
        L₁ : List (Prod α Bool)
        a : α
        b : Bool
        e : List (Prod α Bool)
        a' : α
        b' : Bool
        hL : Eq (List.cons { fst := a, snd := b } L₁) (HAppend.hAppend List.nil (List. …
        ⊢ Or (Exists fun L => And (FreeGroup.Red.Step L₁ L) (Eq (HAppend.hAppend List. …
      -/
    · simp at hL
      /-
        case mp.not.nil
        α : Type u
        L₁ : List (Prod α Bool)
        a : α
        b : Bool
        e : List (Prod α Bool)
        a' : α
        b' : Bool
        hL : And (And (Eq a a') (Eq b b')) (Eq L₁ (List.cons { fst := a', snd := b'.no …
        ⊢ Or (Exists fun L => And (FreeGroup.Red.Step L₁ L) (Eq (HAppend.hAppend List. …
      -/
      simp [*]
      /-
        🎉 no goals
      -/
      /-
        case mp.not.cons
        α : Type u
        L₁ : List (Prod α Bool)
        a : α
        b : Bool
        e : List (Prod α Bool)
        a' : α
        b' : Bool
        p : Prod α Bool
        s' : List (Prod α Bool)
        hL : Eq (List.cons { fst := a, snd := b } L₁) (HAppend.hAppend (List.cons p s' …
        ⊢ Or (Exists fun L => And (FreeGroup.Red.Step L₁ L) (Eq (HAppend.hAppend (List …
      -/
    · simp at hL
      /-
        case mp.not.cons
        α : Type u
        L₁ : List (Prod α Bool)
        a : α
        b : Bool
        e : List (Prod α Bool)
        a' : α
        b' : Bool
        p : Prod α Bool
        s' : List (Prod α Bool)
        hL : And (Eq { fst := a, snd := b } p) (Eq L₁ (HAppend.hAppend s' (List.cons { …
        ⊢ Or (Exists fun L => And (FreeGroup.Red.Step L₁ L) (Eq (HAppend.hAppend (List …
      -/
      rcases hL with ⟨rfl, rfl⟩
      /-
        case mp.not.cons.intro
        α : Type u
        a : α
        b : Bool
        e : List (Prod α Bool)
        a' : α
        b' : Bool
        s' : List (Prod α Bool)
        ⊢ Or (Exists fun L => And (FreeGroup.Red.Step (HAppend.hAppend s' (List.cons { …
      -/
      refine Or.inl ⟨s' ++ e, Step.not, ?_⟩
      /-
        case mp.not.cons.intro
        α : Type u
        a : α
        b : Bool
        e : List (Prod α Bool)
        a' : α
        b' : Bool
        s' : List (Prod α Bool)
        ⊢ Eq (HAppend.hAppend (List.cons { fst := a, snd := b } s') e) (List.cons { fs …
      -/
      simp
      /-
        🎉 no goals
      -/
    /-
      case mpr
      α : Type u
      L₁ L₂ : List (Prod α Bool)
      a : α
      b : Bool
      ⊢ Or (Exists fun L => And (FreeGroup.Red.Step L₁ L) (Eq L₂ (List.cons { fst := …
    -/
  · rintro (⟨L, h, rfl⟩ | rfl)
      /-
        case mpr.inl.intro.intro
        α : Type u
        L₁ : List (Prod α Bool)
        a : α
        b : Bool
        L : List (Prod α Bool)
        h : FreeGroup.Red.Step L₁ L
        ⊢ FreeGroup.Red.Step (List.cons { fst := a, snd := b } L₁) (List.cons { fst := …
      -/
    · exact Step.cons h
      /-
        🎉 no goals
      -/
      /-
        case mpr.inr
        α : Type u
        L₂ : List (Prod α Bool)
        a : α
        b : Bool
        ⊢ FreeGroup.Red.Step (List.cons { fst := a, snd := b } (List.cons { fst := a,  …
      -/
    · exact Step.cons_not
      /-
        🎉 no goals
      -/


@[to_additive]
theorem not_step_singleton : ∀ {p : α × Bool}, ¬Step [p] L
                 /-
                   α : Type u
                   L : List (Prod α Bool)
                   a : α
                   b : Bool
                   ⊢ Not (FreeGroup.Red.Step (List.cons { fst := a, snd := b } List.nil) L)
                 -/
  | (a, b) => by simp [Step.cons_left_iff, not_step_nil]
                 /-
                   🎉 no goals
                 -/


@[to_additive]
theorem Step.cons_cons_iff : ∀ {p : α × Bool}, Step (p :: L₁) (p :: L₂) ↔ Step L₁ L₂ := by
  /-
    α : Type u
    L₁ L₂ : List (Prod α Bool)
    ⊢ ∀ {p : Prod α Bool}, Iff (FreeGroup.Red.Step (List.cons p L₁) (List.cons p L …
  -/
  simp +contextual [Step.cons_left_iff, iff_def, or_imp]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem Step.append_left_iff : ∀ L, Step (L ++ L₁) (L ++ L₂) ↔ Step L₁ L₂
             /-
               α : Type u
               L₁ L₂ : List (Prod α Bool)
               ⊢ Iff (FreeGroup.Red.Step (HAppend.hAppend List.nil L₁) (HAppend.hAppend List. …
             -/
  | [] => by simp
             /-
               🎉 no goals
             -/
                 /-
                   α : Type u
                   L₁ L₂ : List (Prod α Bool)
                   p : Prod α Bool
                   l : List (Prod α Bool)
                   ⊢ Iff (FreeGroup.Red.Step (HAppend.hAppend (List.cons p l) L₁) (HAppend.hAppen …
                 -/
  | p :: l => by simp [Step.append_left_iff l, Step.cons_cons_iff]
                 /-
                   🎉 no goals
                 -/


@[to_additive]
theorem Step.diamond_aux :
    ∀ {L₁ L₂ L₃ L₄ : List (α × Bool)} {x1 b1 x2 b2},
      L₁ ++ (x1, b1) :: (x1, !b1) :: L₂ = L₃ ++ (x2, b2) :: (x2, !b2) :: L₄ →
        L₁ ++ L₂ = L₃ ++ L₄ ∨ ∃ L₅, Red.Step (L₁ ++ L₂) L₅ ∧ Red.Step (L₃ ++ L₄) L₅
                                      /-
                                        α : Type u
                                        x✝⁵ x✝⁴ : List (Prod α Bool)
                                        x✝³ : α
                                        x✝² : Bool
                                        x✝¹ : α
                                        x✝ : Bool
                                        H : Eq (HAppend.hAppend List.nil (List.cons { fst := x✝³, snd := x✝² } (List.c …
                                        ⊢ Or (Eq (HAppend.hAppend List.nil x✝⁵) (HAppend.hAppend List.nil x✝⁴)) (Exist …
                                      -/
  | [], _, [], _, _, _, _, _, H => by injections; subst_vars; simp
                                                              /-
                                                                🎉 no goals
                                                              -/
                                              /-
                                                α : Type u
                                                x✝⁵ : List (Prod α Bool)
                                                x3 : α
                                                b3 : Bool
                                                x✝⁴ : List (Prod α Bool)
                                                x✝³ : α
                                                x✝² : Bool
                                                x✝¹ : α
                                                x✝ : Bool
                                                H : Eq (HAppend.hAppend List.nil (List.cons { fst := x✝³, snd := x✝² } (List.c …
                                                ⊢ Or (Eq (HAppend.hAppend List.nil x✝⁵) (HAppend.hAppend (List.cons { fst := x …
                                              -/
  | [], _, [(x3, b3)], _, _, _, _, _, H => by injections; subst_vars; simp
                                                                      /-
                                                                        🎉 no goals
                                                                      -/
                                              /-
                                                α : Type u
                                                x3 : α
                                                b3 : Bool
                                                x✝⁵ x✝⁴ : List (Prod α Bool)
                                                x✝³ : α
                                                x✝² : Bool
                                                x✝¹ : α
                                                x✝ : Bool
                                                H : Eq (HAppend.hAppend (List.cons { fst := x3, snd := b3 } List.nil) (List.co …
                                                ⊢ Or (Eq (HAppend.hAppend (List.cons { fst := x3, snd := b3 } List.nil) x✝⁵) ( …
                                              -/
  | [(x3, b3)], _, [], _, _, _, _, _, H => by injections; subst_vars; simp
                                                                      /-
                                                                        🎉 no goals
                                                                      -/
  | [], _, (x3, b3) :: (x4, b4) :: tl, _, _, _, _, _, H => by
    /-
      α : Type u
      x✝⁵ : List (Prod α Bool)
      x3 : α
      b3 : Bool
      x4 : α
      b4 : Bool
      tl x✝⁴ : List (Prod α Bool)
      x✝³ : α
      x✝² : Bool
      x✝¹ : α
      x✝ : Bool
      H : Eq (HAppend.hAppend List.nil (List.cons { fst := x✝³, snd := x✝² } (List.c …
      ⊢ Or (Eq (HAppend.hAppend List.nil x✝⁵) (HAppend.hAppend (List.cons { fst := x …
    -/
    injections; subst_vars; right; exact ⟨_, Red.Step.not, Red.Step.cons_not⟩
                                   /-
                                     🎉 no goals
                                   -/
  | (x3, b3) :: (x4, b4) :: tl, _, [], _, _, _, _, _, H => by
    /-
      α : Type u
      x3 : α
      b3 : Bool
      x4 : α
      b4 : Bool
      tl x✝⁵ x✝⁴ : List (Prod α Bool)
      x✝³ : α
      x✝² : Bool
      x✝¹ : α
      x✝ : Bool
      H : Eq (HAppend.hAppend (List.cons { fst := x3, snd := b3 } (List.cons { fst : …
      ⊢ Or (Eq (HAppend.hAppend (List.cons { fst := x3, snd := b3 } (List.cons { fst …
    -/
    injections; subst_vars; right; simpa using ⟨_, Red.Step.cons_not, Red.Step.not⟩
                                   /-
                                     🎉 no goals
                                   -/
  | (x3, b3) :: tl, _, (x4, b4) :: tl2, _, _, _, _, _, H =>
    let ⟨H1, H2⟩ := List.cons.inj H
    match Step.diamond_aux H2 with
                                /-
                                  α : Type u
                                  x3 : α
                                  b3 : Bool
                                  tl x✝⁵ : List (Prod α Bool)
                                  x4 : α
                                  b4 : Bool
                                  tl2 x✝⁴ : List (Prod α Bool)
                                  x✝³ : α
                                  x✝² : Bool
                                  x✝¹ : α
                                  x✝ : Bool
                                  H : Eq (HAppend.hAppend (List.cons { fst := x3, snd := b3 } tl) (List.cons { f …
                                  H1 : Eq { fst := x3, snd := b3 } { fst := x4, snd := b4 }
                                  H2 : Eq (tl.append (List.cons { fst := x✝³, snd := x✝² } (List.cons { fst := x …
                                  H3 : Eq (HAppend.hAppend tl x✝⁵) (HAppend.hAppend tl2 x✝⁴)
                                  ⊢ Eq (HAppend.hAppend (List.cons { fst := x3, snd := b3 } tl) x✝⁵) (HAppend.hA …
                                -/
    | Or.inl H3 => Or.inl <| by simp [H1, H3]
                                /-
                                  🎉 no goals
                                -/
                                                         /-
                                                           α : Type u
                                                           x3 : α
                                                           b3 : Bool
                                                           tl x✝⁵ : List (Prod α Bool)
                                                           x4 : α
                                                           b4 : Bool
                                                           tl2 x✝⁴ : List (Prod α Bool)
                                                           x✝³ : α
                                                           x✝² : Bool
                                                           x✝¹ : α
                                                           x✝ : Bool
                                                           H : Eq (HAppend.hAppend (List.cons { fst := x3, snd := b3 } tl) (List.cons { f …
                                                           H1 : Eq { fst := x3, snd := b3 } { fst := x4, snd := b4 }
                                                           H2 : Eq (tl.append (List.cons { fst := x✝³, snd := x✝² } (List.cons { fst := x …
                                                           L₅ : List (Prod α Bool)
                                                           H3 : FreeGroup.Red.Step (HAppend.hAppend tl x✝⁵) L₅
                                                           H4 : FreeGroup.Red.Step (HAppend.hAppend tl2 x✝⁴) L₅
                                                           ⊢ FreeGroup.Red.Step (HAppend.hAppend (List.cons { fst := x4, snd := b4 } tl2) …
                                                         -/
    | Or.inr ⟨L₅, H3, H4⟩ => Or.inr ⟨_, Step.cons H3, by simpa [H1] using Step.cons H4⟩
                                                         /-
                                                           🎉 no goals
                                                         -/


@[to_additive]
theorem Step.diamond :
    ∀ {L₁ L₂ L₃ L₄ : List (α × Bool)},
      Red.Step L₁ L₃ → Red.Step L₂ L₄ → L₁ = L₂ → L₃ = L₄ ∨ ∃ L₅, Red.Step L₃ L₅ ∧ Red.Step L₄ L₅
  | _, _, _, _, Red.Step.not, Red.Step.not, H => Step.diamond_aux H


@[to_additive]
theorem Step.to_red : Step L₁ L₂ → Red L₁ L₂ :=
  ReflTransGen.single


/-- **Church-Rosser theorem** for word reduction: If `w1 w2 w3` are words such that `w1` reduces
to `w2` and `w3` respectively, then there is a word `w4` such that `w2` and `w3` reduce to `w4`
respectively. This is also known as Newman's diamond lemma. -/
@[to_additive
  "**Church-Rosser theorem** for word reduction: If `w1 w2 w3` are words such that `w1` reduces
  to `w2` and `w3` respectively, then there is a word `w4` such that `w2` and `w3` reduce to `w4`
  respectively. This is also known as Newman's diamond lemma."]
theorem church_rosser : Red L₁ L₂ → Red L₁ L₃ → Join Red L₂ L₃ :=
  Relation.church_rosser fun _ b c hab hac =>
    match b, c, Red.Step.diamond hab hac rfl with
                                 /-
                                   α : Type u
                                   L₁ L₂ L₃ x✝ b✝ c b : List (Prod α Bool)
                                   hab hac : FreeGroup.Red.Step x✝ b
                                   ⊢ Relation.ReflGen FreeGroup.Red.Step b b
                                 -/
                                 /-
                                   🎉 no goals
                                 -/
    | b, _, Or.inl rfl => ⟨b, by rfl, by rfl⟩
                                         /-
                                           🎉 no goals
                                         -/
    | _, _, Or.inr ⟨d, hbd, hcd⟩ => ⟨d, ReflGen.single hbd, hcd.to_red⟩


@[to_additive]
theorem cons_cons {p} : Red L₁ L₂ → Red (p :: L₁) (p :: L₂) :=
  ReflTransGen.lift (List.cons p) fun _ _ => Step.cons


@[to_additive]
theorem cons_cons_iff (p) : Red (p :: L₁) (p :: L₂) ↔ Red L₁ L₂ :=
  Iff.intro
    (by
      /-
        α : Type u
        L₁ L₂ : List (Prod α Bool)
        p : Prod α Bool
        ⊢ FreeGroup.Red (List.cons p L₁) (List.cons p L₂) → FreeGroup.Red L₁ L₂
      -/
      generalize eq₁ : (p :: L₁ : List _) = LL₁
      /-
        α : Type u
        L₁ L₂ : List (Prod α Bool)
        p : Prod α Bool
        LL₁ : List (Prod α Bool)
        eq₁ : Eq (List.cons p L₁) LL₁
        ⊢ FreeGroup.Red LL₁ (List.cons p L₂) → FreeGroup.Red L₁ L₂
      -/
      generalize eq₂ : (p :: L₂ : List _) = LL₂
      /-
        α : Type u
        L₁ L₂ : List (Prod α Bool)
        p : Prod α Bool
        LL₁ : List (Prod α Bool)
        eq₁ : Eq (List.cons p L₁) LL₁
        LL₂ : List (Prod α Bool)
        eq₂ : Eq (List.cons p L₂) LL₂
        ⊢ FreeGroup.Red LL₁ LL₂ → FreeGroup.Red L₁ L₂
      -/
      intro h
      induction' h using Relation.ReflTransGen.head_induction_on
        with L₁ L₂ h₁₂ h ih
        generalizing L₁ L₂
        /-
          case refl
          α : Type u
          p : Prod α Bool
          LL₁ LL₂ L₁ L₂ : List (Prod α Bool)
          eq₁ : Eq (List.cons p L₁) LL₂
          eq₂ : Eq (List.cons p L₂) LL₂
          ⊢ FreeGroup.Red L₁ L₂
        -/
      · subst_vars
        /-
          case refl
          α : Type u
          p : Prod α Bool
          LL₁ L₁ L₂ : List (Prod α Bool)
          eq₂ : Eq (List.cons p L₂) (List.cons p L₁)
          ⊢ FreeGroup.Red L₁ L₂
        -/
        cases eq₂
        /-
          case refl.refl
          α : Type u
          p : Prod α Bool
          LL₁ L₁ : List (Prod α Bool)
          ⊢ FreeGroup.Red L₁ L₁
        -/
        constructor
        /-
          🎉 no goals
        -/
        /-
          case head
          α : Type u
          p : Prod α Bool
          LL₁ LL₂ L₁✝ L₂✝ : List (Prod α Bool)
          h₁₂ : FreeGroup.Red.Step L₁✝ L₂✝
          h : Relation.ReflTransGen FreeGroup.Red.Step L₂✝ LL₂
          ih : ∀ {L₁ L₂ : List (Prod α Bool)}, Eq (List.cons p L₁) L₂✝ → Eq (List.cons p …
          L₁ L₂ : List (Prod α Bool)
          eq₁ : Eq (List.cons p L₁) L₁✝
          eq₂ : Eq (List.cons p L₂) LL₂
          ⊢ FreeGroup.Red L₁ L₂
        -/
      · subst_vars
        /-
          case head
          α : Type u
          p : Prod α Bool
          LL₁ L₂✝ L₁ L₂ : List (Prod α Bool)
          h : Relation.ReflTransGen FreeGroup.Red.Step L₂✝ (List.cons p L₂)
          ih : ∀ {L₁ L₂_1 : List (Prod α Bool)}, Eq (List.cons p L₁) L₂✝ → Eq (List.cons …
          h₁₂ : FreeGroup.Red.Step (List.cons p L₁) L₂✝
          ⊢ FreeGroup.Red L₁ L₂
        -/
        cases' p with a b
        /-
          case head.mk
          α : Type u
          LL₁ L₂✝ L₁ L₂ : List (Prod α Bool)
          a : α
          b : Bool
          h : Relation.ReflTransGen FreeGroup.Red.Step L₂✝ (List.cons { fst := a, snd := …
          ih : ∀ {L₁ L₂_1 : List (Prod α Bool)}, Eq (List.cons { fst := a, snd := b } L₁ …
          h₁₂ : FreeGroup.Red.Step (List.cons { fst := a, snd := b } L₁) L₂✝
          ⊢ FreeGroup.Red L₁ L₂
        -/
        rw [Step.cons_left_iff] at h₁₂
        /-
          case head.mk
          α : Type u
          LL₁ L₂✝ L₁ L₂ : List (Prod α Bool)
          a : α
          b : Bool
          h : Relation.ReflTransGen FreeGroup.Red.Step L₂✝ (List.cons { fst := a, snd := …
          ih : ∀ {L₁ L₂_1 : List (Prod α Bool)}, Eq (List.cons { fst := a, snd := b } L₁ …
          h₁₂ : Or (Exists fun L => And (FreeGroup.Red.Step L₁ L) (Eq L₂✝ (List.cons { f …
          ⊢ FreeGroup.Red L₁ L₂
        -/
        rcases h₁₂ with (⟨L, h₁₂, rfl⟩ | rfl)
          /-
            case head.mk.inl.intro.intro
            α : Type u
            LL₁ L₁ L₂ : List (Prod α Bool)
            a : α
            b : Bool
            L : List (Prod α Bool)
            h₁₂ : FreeGroup.Red.Step L₁ L
            h : Relation.ReflTransGen FreeGroup.Red.Step (List.cons { fst := a, snd := b } …
            ih : ∀ {L₁ L₂_1 : List (Prod α Bool)}, Eq (List.cons { fst := a, snd := b } L₁ …
            ⊢ FreeGroup.Red L₁ L₂
          -/
        · exact (ih rfl rfl).head h₁₂
          /-
            🎉 no goals
          -/
          /-
            case head.mk.inr
            α : Type u
            LL₁ L₂✝ L₂ : List (Prod α Bool)
            a : α
            b : Bool
            h : Relation.ReflTransGen FreeGroup.Red.Step L₂✝ (List.cons { fst := a, snd := …
            ih : ∀ {L₁ L₂_1 : List (Prod α Bool)}, Eq (List.cons { fst := a, snd := b } L₁ …
            ⊢ FreeGroup.Red (List.cons { fst := a, snd := b.not } L₂✝) L₂
          -/
        · exact (cons_cons h).tail Step.cons_not_rev)
          /-
            🎉 no goals
          -/
    cons_cons


@[to_additive]
theorem append_append_left_iff : ∀ L, Red (L ++ L₁) (L ++ L₂) ↔ Red L₁ L₂
  | [] => Iff.rfl
                 /-
                   α : Type u
                   L₁ L₂ : List (Prod α Bool)
                   p : Prod α Bool
                   L : List (Prod α Bool)
                   ⊢ Iff (FreeGroup.Red (HAppend.hAppend (List.cons p L) L₁) (HAppend.hAppend (Li …
                 -/
  | p :: L => by simp [append_append_left_iff L, cons_cons_iff]
                 /-
                   🎉 no goals
                 -/


@[to_additive]
theorem append_append (h₁ : Red L₁ L₃) (h₂ : Red L₂ L₄) : Red (L₁ ++ L₂) (L₃ ++ L₄) :=
  (h₁.lift (fun L => L ++ L₂) fun _ _ => Step.append_right).trans ((append_append_left_iff _).2 h₂)


@[to_additive]
theorem to_append_iff : Red L (L₁ ++ L₂) ↔ ∃ L₃ L₄, L = L₃ ++ L₄ ∧ Red L₃ L₁ ∧ Red L₄ L₂ :=
  Iff.intro
    (by
      /-
        α : Type u
        L L₁ L₂ : List (Prod α Bool)
        ⊢ FreeGroup.Red L (HAppend.hAppend L₁ L₂) → Exists fun L₃ => Exists fun L₄ =>  …
      -/
      generalize eq : L₁ ++ L₂ = L₁₂
      /-
        α : Type u
        L L₁ L₂ L₁₂ : List (Prod α Bool)
        eq : Eq (HAppend.hAppend L₁ L₂) L₁₂
        ⊢ FreeGroup.Red L L₁₂ → Exists fun L₃ => Exists fun L₄ => And (Eq L (HAppend.h …
      -/
      intro h
      /-
        α : Type u
        L L₁ L₂ L₁₂ : List (Prod α Bool)
        eq : Eq (HAppend.hAppend L₁ L₂) L₁₂
        h : FreeGroup.Red L L₁₂
        ⊢ Exists fun L₃ => Exists fun L₄ => And (Eq L (HAppend.hAppend L₃ L₄)) (And (F …
      -/
      induction' h with L' L₁₂ hLL' h ih generalizing L₁ L₂
        /-
          case refl
          α : Type u
          L L₁₂ L₁ L₂ : List (Prod α Bool)
          eq : Eq (HAppend.hAppend L₁ L₂) L
          ⊢ Exists fun L₃ => Exists fun L₄ => And (Eq L (HAppend.hAppend L₃ L₄)) (And (F …
        -/
      · exact ⟨_, _, eq.symm, by rfl, by rfl⟩
        /-
          🎉 no goals
        -/
        /-
          case tail
          α : Type u
          L L₁₂✝ L' L₁₂ : List (Prod α Bool)
          hLL' : Relation.ReflTransGen FreeGroup.Red.Step L L'
          h : FreeGroup.Red.Step L' L₁₂
          ih : ∀ {L₁ L₂ : List (Prod α Bool)}, Eq (HAppend.hAppend L₁ L₂) L' → Exists fu …
          L₁ L₂ : List (Prod α Bool)
          eq : Eq (HAppend.hAppend L₁ L₂) L₁₂
          ⊢ Exists fun L₃ => Exists fun L₄ => And (Eq L (HAppend.hAppend L₃ L₄)) (And (F …
        -/
      · cases' h with s e a b
        /-
          case tail.not
          α : Type u
          L L₁₂ L₁ L₂ s e : List (Prod α Bool)
          a : α
          b : Bool
          hLL' : Relation.ReflTransGen FreeGroup.Red.Step L (HAppend.hAppend s (List.con …
          ih : ∀ {L₁ L₂ : List (Prod α Bool)}, Eq (HAppend.hAppend L₁ L₂) (HAppend.hAppe …
          eq : Eq (HAppend.hAppend L₁ L₂) (HAppend.hAppend s e)
          ⊢ Exists fun L₃ => Exists fun L₄ => And (Eq L (HAppend.hAppend L₃ L₄)) (And (F …
        -/
        rcases List.append_eq_append_iff.1 eq with (⟨s', rfl, rfl⟩ | ⟨e', rfl, rfl⟩)
        · have : L₁ ++ (s' ++ (a, b) :: (a, not b) :: e) = L₁ ++ s' ++ (a, b) :: (a, not b) :: e :=
            by simp
          /-
            case tail.not.inl.intro.intro
            α : Type u
            L L₁₂ L₁ e : List (Prod α Bool)
            a : α
            b : Bool
            s' : List (Prod α Bool)
            hLL' : Relation.ReflTransGen FreeGroup.Red.Step L (HAppend.hAppend (HAppend.hA …
            ih : ∀ {L₁_1 L₂ : List (Prod α Bool)}, Eq (HAppend.hAppend L₁_1 L₂) (HAppend.h …
            eq : Eq (HAppend.hAppend L₁ (HAppend.hAppend s' e)) (HAppend.hAppend (HAppend. …
            this : Eq (HAppend.hAppend L₁ (HAppend.hAppend s' (List.cons { fst := a, snd : …
            ⊢ Exists fun L₃ => Exists fun L₄ => And (Eq L (HAppend.hAppend L₃ L₄)) (And (F …
          -/
          rcases ih this with ⟨w₁, w₂, rfl, h₁, h₂⟩
          /-
            case tail.not.inl.intro.intro.intro.intro.intro.intro
            α : Type u
            L₁₂ L₁ e : List (Prod α Bool)
            a : α
            b : Bool
            s' : List (Prod α Bool)
            eq : Eq (HAppend.hAppend L₁ (HAppend.hAppend s' e)) (HAppend.hAppend (HAppend. …
            this : Eq (HAppend.hAppend L₁ (HAppend.hAppend s' (List.cons { fst := a, snd : …
            w₁ w₂ : List (Prod α Bool)
            hLL' : Relation.ReflTransGen FreeGroup.Red.Step (HAppend.hAppend w₁ w₂) (HAppe …
            ih : ∀ {L₁_1 L₂ : List (Prod α Bool)}, Eq (HAppend.hAppend L₁_1 L₂) (HAppend.h …
            h₁ : FreeGroup.Red w₁ L₁
            h₂ : FreeGroup.Red w₂ (HAppend.hAppend s' (List.cons { fst := a, snd := b } (L …
            ⊢ Exists fun L₃ => Exists fun L₄ => And (Eq (HAppend.hAppend w₁ w₂) (HAppend.h …
          -/
          exact ⟨w₁, w₂, rfl, h₁, h₂.tail Step.not⟩
          /-
            🎉 no goals
          -/
        · have : s ++ (a, b) :: (a, not b) :: e' ++ L₂ = s ++ (a, b) :: (a, not b) :: (e' ++ L₂) :=
            by simp
          /-
            case tail.not.inr.intro.intro
            α : Type u
            L L₁₂ L₂ s : List (Prod α Bool)
            a : α
            b : Bool
            e' : List (Prod α Bool)
            hLL' : Relation.ReflTransGen FreeGroup.Red.Step L (HAppend.hAppend s (List.con …
            ih : ∀ {L₁ L₂_1 : List (Prod α Bool)}, Eq (HAppend.hAppend L₁ L₂_1) (HAppend.h …
            eq : Eq (HAppend.hAppend (HAppend.hAppend s e') L₂) (HAppend.hAppend s (HAppen …
            this : Eq (HAppend.hAppend (HAppend.hAppend s (List.cons { fst := a, snd := b  …
            ⊢ Exists fun L₃ => Exists fun L₄ => And (Eq L (HAppend.hAppend L₃ L₄)) (And (F …
          -/
          rcases ih this with ⟨w₁, w₂, rfl, h₁, h₂⟩
          /-
            case tail.not.inr.intro.intro.intro.intro.intro.intro
            α : Type u
            L₁₂ L₂ s : List (Prod α Bool)
            a : α
            b : Bool
            e' : List (Prod α Bool)
            eq : Eq (HAppend.hAppend (HAppend.hAppend s e') L₂) (HAppend.hAppend s (HAppen …
            this : Eq (HAppend.hAppend (HAppend.hAppend s (List.cons { fst := a, snd := b  …
            w₁ w₂ : List (Prod α Bool)
            hLL' : Relation.ReflTransGen FreeGroup.Red.Step (HAppend.hAppend w₁ w₂) (HAppe …
            ih : ∀ {L₁ L₂_1 : List (Prod α Bool)}, Eq (HAppend.hAppend L₁ L₂_1) (HAppend.h …
            h₁ : FreeGroup.Red w₁ (HAppend.hAppend s (List.cons { fst := a, snd := b } (Li …
            h₂ : FreeGroup.Red w₂ L₂
            ⊢ Exists fun L₃ => Exists fun L₄ => And (Eq (HAppend.hAppend w₁ w₂) (HAppend.h …
          -/
          exact ⟨w₁, w₂, rfl, h₁.tail Step.not, h₂⟩)
          /-
            🎉 no goals
          -/
    fun ⟨_, _, Eq, h₃, h₄⟩ => Eq.symm ▸ append_append h₃ h₄


/-- The empty word `[]` only reduces to itself. -/
@[to_additive "The empty word `[]` only reduces to itself."]
theorem nil_iff : Red [] L ↔ L = [] :=
  reflTransGen_iff_eq fun _ => Red.not_step_nil


/-- A letter only reduces to itself. -/
@[to_additive "A letter only reduces to itself."]
theorem singleton_iff {x} : Red [x] L₁ ↔ L₁ = [x] :=
  reflTransGen_iff_eq fun _ => not_step_singleton


/-- If `x` is a letter and `w` is a word such that `xw` reduces to the empty word, then `w` reduces
to `x⁻¹` -/
@[to_additive
  "If `x` is a letter and `w` is a word such that `x + w` reduces to the empty word, then `w`
  reduces to `-x`."]
theorem cons_nil_iff_singleton {x b} : Red ((x, b) :: L) [] ↔ Red L [(x, not b)] :=
  Iff.intro
    (fun h => by
      /-
        α : Type u
        L : List (Prod α Bool)
        x : α
        b : Bool
        h : FreeGroup.Red (List.cons { fst := x, snd := b } L) List.nil
        ⊢ FreeGroup.Red L (List.cons { fst := x, snd := b.not } List.nil)
      -/
      have h₁ : Red ((x, not b) :: (x, b) :: L) [(x, not b)] := cons_cons h
      /-
        α : Type u
        L : List (Prod α Bool)
        x : α
        b : Bool
        h : FreeGroup.Red (List.cons { fst := x, snd := b } L) List.nil
        h₁ : FreeGroup.Red (List.cons { fst := x, snd := b.not } (List.cons { fst := x …
        ⊢ FreeGroup.Red L (List.cons { fst := x, snd := b.not } List.nil)
      -/
      have h₂ : Red ((x, not b) :: (x, b) :: L) L := ReflTransGen.single Step.cons_not_rev
      /-
        α : Type u
        L : List (Prod α Bool)
        x : α
        b : Bool
        h : FreeGroup.Red (List.cons { fst := x, snd := b } L) List.nil
        h₁ : FreeGroup.Red (List.cons { fst := x, snd := b.not } (List.cons { fst := x …
        h₂ : FreeGroup.Red (List.cons { fst := x, snd := b.not } (List.cons { fst := x …
        ⊢ FreeGroup.Red L (List.cons { fst := x, snd := b.not } List.nil)
      -/
      let ⟨L', h₁, h₂⟩ := church_rosser h₁ h₂
      /-
        α : Type u
        L : List (Prod α Bool)
        x : α
        b : Bool
        h : FreeGroup.Red (List.cons { fst := x, snd := b } L) List.nil
        h₁✝ : FreeGroup.Red (List.cons { fst := x, snd := b.not } (List.cons { fst :=  …
        h₂✝ : FreeGroup.Red (List.cons { fst := x, snd := b.not } (List.cons { fst :=  …
        L' : List (Prod α Bool)
        h₁ : FreeGroup.Red (List.cons { fst := x, snd := b.not } List.nil) L'
        h₂ : FreeGroup.Red L L'
        ⊢ FreeGroup.Red L (List.cons { fst := x, snd := b.not } List.nil)
      -/
      rw [singleton_iff] at h₁
      /-
        α : Type u
        L : List (Prod α Bool)
        x : α
        b : Bool
        h : FreeGroup.Red (List.cons { fst := x, snd := b } L) List.nil
        h₁✝ : FreeGroup.Red (List.cons { fst := x, snd := b.not } (List.cons { fst :=  …
        h₂✝ : FreeGroup.Red (List.cons { fst := x, snd := b.not } (List.cons { fst :=  …
        L' : List (Prod α Bool)
        h₁ : Eq L' (List.cons { fst := x, snd := b.not } List.nil)
        h₂ : FreeGroup.Red L L'
        ⊢ FreeGroup.Red L (List.cons { fst := x, snd := b.not } List.nil)
      -/
      subst L'
      /-
        α : Type u
        L : List (Prod α Bool)
        x : α
        b : Bool
        h : FreeGroup.Red (List.cons { fst := x, snd := b } L) List.nil
        h₁ : FreeGroup.Red (List.cons { fst := x, snd := b.not } (List.cons { fst := x …
        h₂✝ : FreeGroup.Red (List.cons { fst := x, snd := b.not } (List.cons { fst :=  …
        h₂ : FreeGroup.Red L (List.cons { fst := x, snd := b.not } List.nil)
        ⊢ FreeGroup.Red L (List.cons { fst := x, snd := b.not } List.nil)
      -/
      assumption)
      /-
        🎉 no goals
      -/
    fun h => (cons_cons h).tail Step.cons_not


@[to_additive]
theorem red_iff_irreducible {x1 b1 x2 b2} (h : (x1, b1) ≠ (x2, b2)) :
    Red [(x1, !b1), (x2, b2)] L ↔ L = [(x1, !b1), (x2, b2)] := by
  /-
    α : Type u
    L : List (Prod α Bool)
    x1 : α
    b1 : Bool
    x2 : α
    b2 : Bool
    h : Ne { fst := x1, snd := b1 } { fst := x2, snd := b2 }
    ⊢ Iff (FreeGroup.Red (List.cons { fst := x1, snd := b1.not } (List.cons { fst  …
  -/
  apply reflTransGen_iff_eq
  /-
    case h
    α : Type u
    L : List (Prod α Bool)
    x1 : α
    b1 : Bool
    x2 : α
    b2 : Bool
    h : Ne { fst := x1, snd := b1 } { fst := x2, snd := b2 }
    ⊢ ∀ (b : List (Prod α Bool)), Not (FreeGroup.Red.Step (List.cons { fst := x1,  …
  -/
  generalize eq : [(x1, not b1), (x2, b2)] = L'
  /-
    case h
    α : Type u
    L : List (Prod α Bool)
    x1 : α
    b1 : Bool
    x2 : α
    b2 : Bool
    h : Ne { fst := x1, snd := b1 } { fst := x2, snd := b2 }
    L' : List (Prod α Bool)
    eq : Eq (List.cons { fst := x1, snd := b1.not } (List.cons { fst := x2, snd := …
    ⊢ ∀ (b : List (Prod α Bool)), Not (FreeGroup.Red.Step L' b)
  -/
  intro L h'
  /-
    case h
    α : Type u
    L✝ : List (Prod α Bool)
    x1 : α
    b1 : Bool
    x2 : α
    b2 : Bool
    h : Ne { fst := x1, snd := b1 } { fst := x2, snd := b2 }
    L' : List (Prod α Bool)
    eq : Eq (List.cons { fst := x1, snd := b1.not } (List.cons { fst := x2, snd := …
    L : List (Prod α Bool)
    h' : FreeGroup.Red.Step L' L
    ⊢ False
  -/
  cases h'
  simp only [List.cons_eq_append_iff, List.cons.injEq, Prod.mk.injEq, and_false,
    List.nil_eq_append_iff, exists_const, or_self, or_false, List.cons_ne_nil] at eq
  /-
    case h.not
    α : Type u
    L : List (Prod α Bool)
    x1 : α
    b1 : Bool
    x2 : α
    b2 : Bool
    h : Ne { fst := x1, snd := b1 } { fst := x2, snd := b2 }
    L₁✝ L₂✝ : List (Prod α Bool)
    x✝ : α
    b✝ : Bool
    eq : And (Eq L₁✝ List.nil) (And (And (Eq x✝ x1) (Eq b✝ b1.not)) (And (And (Eq  …
    ⊢ False
  -/
  rcases eq with ⟨rfl, ⟨rfl, rfl⟩, ⟨rfl, rfl⟩, rfl⟩
  /-
    case h.not.intro.intro.intro.intro.intro
    α : Type u
    L : List (Prod α Bool)
    b1 : Bool
    x✝ : α
    h : Ne { fst := x✝, snd := b1 } { fst := x✝, snd := b1.not.not }
    ⊢ False
  -/
  simp at h
  /-
    🎉 no goals
  -/


/-- If `x` and `y` are distinct letters and `w₁ w₂` are words such that `xw₁` reduces to `yw₂`, then
`w₁` reduces to `x⁻¹yw₂`. -/
@[to_additive "If `x` and `y` are distinct letters and `w₁ w₂` are words such that `x + w₁` reduces
  to `y + w₂`, then `w₁` reduces to `-x + y + w₂`."]
theorem inv_of_red_of_ne {x1 b1 x2 b2} (H1 : (x1, b1) ≠ (x2, b2))
    (H2 : Red ((x1, b1) :: L₁) ((x2, b2) :: L₂)) : Red L₁ ((x1, not b1) :: (x2, b2) :: L₂) := by
  /-
    α : Type u
    L₁ L₂ : List (Prod α Bool)
    x1 : α
    b1 : Bool
    x2 : α
    b2 : Bool
    H1 : Ne { fst := x1, snd := b1 } { fst := x2, snd := b2 }
    H2 : FreeGroup.Red (List.cons { fst := x1, snd := b1 } L₁) (List.cons { fst := …
    ⊢ FreeGroup.Red L₁ (List.cons { fst := x1, snd := b1.not } (List.cons { fst := …
  -/
  have : Red ((x1, b1) :: L₁) ([(x2, b2)] ++ L₂) := H2
  /-
    α : Type u
    L₁ L₂ : List (Prod α Bool)
    x1 : α
    b1 : Bool
    x2 : α
    b2 : Bool
    H1 : Ne { fst := x1, snd := b1 } { fst := x2, snd := b2 }
    H2 : FreeGroup.Red (List.cons { fst := x1, snd := b1 } L₁) (List.cons { fst := …
    this : FreeGroup.Red (List.cons { fst := x1, snd := b1 } L₁) (HAppend.hAppend  …
    ⊢ FreeGroup.Red L₁ (List.cons { fst := x1, snd := b1.not } (List.cons { fst := …
  -/
  rcases to_append_iff.1 this with ⟨_ | ⟨p, L₃⟩, L₄, eq, h₁, h₂⟩
    /-
      case intro.nil.intro.intro.intro
      α : Type u
      L₁ L₂ : List (Prod α Bool)
      x1 : α
      b1 : Bool
      x2 : α
      b2 : Bool
      H1 : Ne { fst := x1, snd := b1 } { fst := x2, snd := b2 }
      H2 : FreeGroup.Red (List.cons { fst := x1, snd := b1 } L₁) (List.cons { fst := …
      this : FreeGroup.Red (List.cons { fst := x1, snd := b1 } L₁) (HAppend.hAppend  …
      L₄ : List (Prod α Bool)
      eq : Eq (List.cons { fst := x1, snd := b1 } L₁) (HAppend.hAppend List.nil L₄)
      h₁ : FreeGroup.Red List.nil (List.cons { fst := x2, snd := b2 } List.nil)
      h₂ : FreeGroup.Red L₄ L₂
      ⊢ FreeGroup.Red L₁ (List.cons { fst := x1, snd := b1.not } (List.cons { fst := …
    -/
  · simp [nil_iff] at h₁
    /-
      🎉 no goals
    -/
    /-
      case intro.cons.intro.intro.intro
      α : Type u
      L₁ L₂ : List (Prod α Bool)
      x1 : α
      b1 : Bool
      x2 : α
      b2 : Bool
      H1 : Ne { fst := x1, snd := b1 } { fst := x2, snd := b2 }
      H2 : FreeGroup.Red (List.cons { fst := x1, snd := b1 } L₁) (List.cons { fst := …
      this : FreeGroup.Red (List.cons { fst := x1, snd := b1 } L₁) (HAppend.hAppend  …
      p : Prod α Bool
      L₃ L₄ : List (Prod α Bool)
      eq : Eq (List.cons { fst := x1, snd := b1 } L₁) (HAppend.hAppend (List.cons p  …
      h₁ : FreeGroup.Red (List.cons p L₃) (List.cons { fst := x2, snd := b2 } List.n …
      h₂ : FreeGroup.Red L₄ L₂
      ⊢ FreeGroup.Red L₁ (List.cons { fst := x1, snd := b1.not } (List.cons { fst := …
    -/
  · cases eq
    /-
      case intro.cons.intro.intro.intro.refl
      α : Type u
      L₂ : List (Prod α Bool)
      x1 : α
      b1 : Bool
      x2 : α
      b2 : Bool
      H1 : Ne { fst := x1, snd := b1 } { fst := x2, snd := b2 }
      L₃ L₄ : List (Prod α Bool)
      h₂ : FreeGroup.Red L₄ L₂
      h₁ : FreeGroup.Red (List.cons { fst := x1, snd := b1 } L₃) (List.cons { fst := …
      H2 : FreeGroup.Red (List.cons { fst := x1, snd := b1 } (L₃.append L₄)) (List.c …
      this : FreeGroup.Red (List.cons { fst := x1, snd := b1 } (L₃.append L₄)) (HApp …
      ⊢ FreeGroup.Red (L₃.append L₄) (List.cons { fst := x1, snd := b1.not } (List.c …
    -/
    show Red (L₃ ++ L₄) ([(x1, not b1), (x2, b2)] ++ L₂)
    /-
      case intro.cons.intro.intro.intro.refl
      α : Type u
      L₂ : List (Prod α Bool)
      x1 : α
      b1 : Bool
      x2 : α
      b2 : Bool
      H1 : Ne { fst := x1, snd := b1 } { fst := x2, snd := b2 }
      L₃ L₄ : List (Prod α Bool)
      h₂ : FreeGroup.Red L₄ L₂
      h₁ : FreeGroup.Red (List.cons { fst := x1, snd := b1 } L₃) (List.cons { fst := …
      H2 : FreeGroup.Red (List.cons { fst := x1, snd := b1 } (L₃.append L₄)) (List.c …
      this : FreeGroup.Red (List.cons { fst := x1, snd := b1 } (L₃.append L₄)) (HApp …
      ⊢ FreeGroup.Red (HAppend.hAppend L₃ L₄) (HAppend.hAppend (List.cons { fst := x …
    -/
    apply append_append _ h₂
    /-
      α : Type u
      L₂ : List (Prod α Bool)
      x1 : α
      b1 : Bool
      x2 : α
      b2 : Bool
      H1 : Ne { fst := x1, snd := b1 } { fst := x2, snd := b2 }
      L₃ L₄ : List (Prod α Bool)
      h₂ : FreeGroup.Red L₄ L₂
      h₁ : FreeGroup.Red (List.cons { fst := x1, snd := b1 } L₃) (List.cons { fst := …
      H2 : FreeGroup.Red (List.cons { fst := x1, snd := b1 } (L₃.append L₄)) (List.c …
      this : FreeGroup.Red (List.cons { fst := x1, snd := b1 } (L₃.append L₄)) (HApp …
      ⊢ FreeGroup.Red L₃ (List.cons { fst := x1, snd := b1.not } (List.cons { fst := …
    -/
    have h₁ : Red ((x1, not b1) :: (x1, b1) :: L₃) [(x1, not b1), (x2, b2)] := cons_cons h₁
    /-
      α : Type u
      L₂ : List (Prod α Bool)
      x1 : α
      b1 : Bool
      x2 : α
      b2 : Bool
      H1 : Ne { fst := x1, snd := b1 } { fst := x2, snd := b2 }
      L₃ L₄ : List (Prod α Bool)
      h₂ : FreeGroup.Red L₄ L₂
      h₁✝ : FreeGroup.Red (List.cons { fst := x1, snd := b1 } L₃) (List.cons { fst : …
      H2 : FreeGroup.Red (List.cons { fst := x1, snd := b1 } (L₃.append L₄)) (List.c …
      this : FreeGroup.Red (List.cons { fst := x1, snd := b1 } (L₃.append L₄)) (HApp …
      h₁ : FreeGroup.Red (List.cons { fst := x1, snd := b1.not } (List.cons { fst := …
      ⊢ FreeGroup.Red L₃ (List.cons { fst := x1, snd := b1.not } (List.cons { fst := …
    -/
    have h₂ : Red ((x1, not b1) :: (x1, b1) :: L₃) L₃ := Step.cons_not_rev.to_red
    /-
      α : Type u
      L₂ : List (Prod α Bool)
      x1 : α
      b1 : Bool
      x2 : α
      b2 : Bool
      H1 : Ne { fst := x1, snd := b1 } { fst := x2, snd := b2 }
      L₃ L₄ : List (Prod α Bool)
      h₂✝ : FreeGroup.Red L₄ L₂
      h₁✝ : FreeGroup.Red (List.cons { fst := x1, snd := b1 } L₃) (List.cons { fst : …
      H2 : FreeGroup.Red (List.cons { fst := x1, snd := b1 } (L₃.append L₄)) (List.c …
      this : FreeGroup.Red (List.cons { fst := x1, snd := b1 } (L₃.append L₄)) (HApp …
      h₁ : FreeGroup.Red (List.cons { fst := x1, snd := b1.not } (List.cons { fst := …
      h₂ : FreeGroup.Red (List.cons { fst := x1, snd := b1.not } (List.cons { fst := …
      ⊢ FreeGroup.Red L₃ (List.cons { fst := x1, snd := b1.not } (List.cons { fst := …
    -/
    rcases church_rosser h₁ h₂ with ⟨L', h₁, h₂⟩
    /-
      case intro.intro
      α : Type u
      L₂ : List (Prod α Bool)
      x1 : α
      b1 : Bool
      x2 : α
      b2 : Bool
      H1 : Ne { fst := x1, snd := b1 } { fst := x2, snd := b2 }
      L₃ L₄ : List (Prod α Bool)
      h₂✝¹ : FreeGroup.Red L₄ L₂
      h₁✝¹ : FreeGroup.Red (List.cons { fst := x1, snd := b1 } L₃) (List.cons { fst  …
      H2 : FreeGroup.Red (List.cons { fst := x1, snd := b1 } (L₃.append L₄)) (List.c …
      this : FreeGroup.Red (List.cons { fst := x1, snd := b1 } (L₃.append L₄)) (HApp …
      h₁✝ : FreeGroup.Red (List.cons { fst := x1, snd := b1.not } (List.cons { fst : …
      h₂✝ : FreeGroup.Red (List.cons { fst := x1, snd := b1.not } (List.cons { fst : …
      L' : List (Prod α Bool)
      h₁ : FreeGroup.Red (List.cons { fst := x1, snd := b1.not } (List.cons { fst := …
      h₂ : FreeGroup.Red L₃ L'
      ⊢ FreeGroup.Red L₃ (List.cons { fst := x1, snd := b1.not } (List.cons { fst := …
    -/
    rw [red_iff_irreducible H1] at h₁
    /-
      case intro.intro
      α : Type u
      L₂ : List (Prod α Bool)
      x1 : α
      b1 : Bool
      x2 : α
      b2 : Bool
      H1 : Ne { fst := x1, snd := b1 } { fst := x2, snd := b2 }
      L₃ L₄ : List (Prod α Bool)
      h₂✝¹ : FreeGroup.Red L₄ L₂
      h₁✝¹ : FreeGroup.Red (List.cons { fst := x1, snd := b1 } L₃) (List.cons { fst  …
      H2 : FreeGroup.Red (List.cons { fst := x1, snd := b1 } (L₃.append L₄)) (List.c …
      this : FreeGroup.Red (List.cons { fst := x1, snd := b1 } (L₃.append L₄)) (HApp …
      h₁✝ : FreeGroup.Red (List.cons { fst := x1, snd := b1.not } (List.cons { fst : …
      h₂✝ : FreeGroup.Red (List.cons { fst := x1, snd := b1.not } (List.cons { fst : …
      L' : List (Prod α Bool)
      h₁ : Eq L' (List.cons { fst := x1, snd := b1.not } (List.cons { fst := x2, snd …
      h₂ : FreeGroup.Red L₃ L'
      ⊢ FreeGroup.Red L₃ (List.cons { fst := x1, snd := b1.not } (List.cons { fst := …
    -/
    rwa [h₁] at h₂
    /-
      🎉 no goals
    -/


@[to_additive]
theorem Step.sublist (H : Red.Step L₁ L₂) : L₂ <+ L₁ := by
  /-
    α : Type u
    L₁ L₂ : List (Prod α Bool)
    H : FreeGroup.Red.Step L₁ L₂
    ⊢ L₂.Sublist L₁
  -/
  cases H; simp
           /-
             🎉 no goals
           -/


/-- If `w₁ w₂` are words such that `w₁` reduces to `w₂`, then `w₂` is a sublist of `w₁`. -/
@[to_additive "If `w₁ w₂` are words such that `w₁` reduces to `w₂`, then `w₂` is a sublist of
  `w₁`."]
protected theorem sublist : Red L₁ L₂ → L₂ <+ L₁ :=
  @reflTransGen_of_transitive_reflexive
    _ (fun a b => b <+ a) _ _ _
    (fun l => List.Sublist.refl l)
    (fun _a _b _c hab hbc => List.Sublist.trans hbc hab)
    (fun _ _ => Red.Step.sublist)


@[to_additive]
theorem length_le (h : Red L₁ L₂) : L₂.length ≤ L₁.length :=
  h.sublist.length_le



@[to_additive]
theorem sizeof_of_step : ∀ {L₁ L₂ : List (α × Bool)},
    Step L₁ L₂ → sizeOf L₂ < sizeOf L₁
  | _, _, @Step.not _ L1 L2 x b => by
    induction L1 with
    | nil =>
      dsimp
      omega
    | cons hd tl ih =>
      dsimp
      exact Nat.add_lt_add_left ih _


@[to_additive]
theorem length (h : Red L₁ L₂) : ∃ n, L₁.length = L₂.length + 2 * n := by
  /-
    α : Type u
    L₁ L₂ : List (Prod α Bool)
    h : FreeGroup.Red L₁ L₂
    ⊢ Exists fun n => Eq L₁.length (HAdd.hAdd L₂.length (HMul.hMul 2 n))
  -/
  induction' h with L₂ L₃ _h₁₂ h₂₃ ih
    /-
      case refl
      α : Type u
      L₁ L₂ : List (Prod α Bool)
      ⊢ Exists fun n => Eq L₁.length (HAdd.hAdd L₁.length (HMul.hMul 2 n))
    -/
  · exact ⟨0, rfl⟩
    /-
      🎉 no goals
    -/
    /-
      case tail
      α : Type u
      L₁ L₂✝ L₂ L₃ : List (Prod α Bool)
      _h₁₂ : Relation.ReflTransGen FreeGroup.Red.Step L₁ L₂
      h₂₃ : FreeGroup.Red.Step L₂ L₃
      ih : Exists fun n => Eq L₁.length (HAdd.hAdd L₂.length (HMul.hMul 2 n))
      ⊢ Exists fun n => Eq L₁.length (HAdd.hAdd L₃.length (HMul.hMul 2 n))
    -/
  · rcases ih with ⟨n, eq⟩
    /-
      case tail.intro
      α : Type u
      L₁ L₂✝ L₂ L₃ : List (Prod α Bool)
      _h₁₂ : Relation.ReflTransGen FreeGroup.Red.Step L₁ L₂
      h₂₃ : FreeGroup.Red.Step L₂ L₃
      n : Nat
      eq : Eq L₁.length (HAdd.hAdd L₂.length (HMul.hMul 2 n))
      ⊢ Exists fun n => Eq L₁.length (HAdd.hAdd L₃.length (HMul.hMul 2 n))
    -/
    exists 1 + n
    /-
      case tail.intro
      α : Type u
      L₁ L₂✝ L₂ L₃ : List (Prod α Bool)
      _h₁₂ : Relation.ReflTransGen FreeGroup.Red.Step L₁ L₂
      h₂₃ : FreeGroup.Red.Step L₂ L₃
      n : Nat
      eq : Eq L₁.length (HAdd.hAdd L₂.length (HMul.hMul 2 n))
      ⊢ Eq L₁.length (HAdd.hAdd L₃.length (HMul.hMul 2 (HAdd.hAdd 1 n)))
    -/
    simp [Nat.mul_add, eq, (Step.length h₂₃).symm, add_assoc]
    /-
      🎉 no goals
    -/


@[to_additive]
theorem antisymm (h₁₂ : Red L₁ L₂) (h₂₁ : Red L₂ L₁) : L₁ = L₂ :=
  h₂₁.sublist.antisymm h₁₂.sublist


@[to_additive FreeAddGroup.equivalence_join_red]
theorem equivalence_join_red : Equivalence (Join (@Red α)) :=
  equivalence_join_reflTransGen fun _ b c hab hac =>
    match b, c, Red.Step.diamond hab hac rfl with
                                 /-
                                   α : Type u
                                   x✝ b✝ c b : List (Prod α Bool)
                                   hab hac : FreeGroup.Red.Step x✝ b
                                   ⊢ Relation.ReflGen FreeGroup.Red.Step b b
                                 -/
                                 /-
                                   🎉 no goals
                                 -/
    | b, _, Or.inl rfl => ⟨b, by rfl, by rfl⟩
                                         /-
                                           🎉 no goals
                                         -/
    | _, _, Or.inr ⟨d, hbd, hcd⟩ => ⟨d, ReflGen.single hbd, ReflTransGen.single hcd⟩


@[to_additive FreeAddGroup.join_red_of_step]
theorem join_red_of_step (h : Red.Step L₁ L₂) : Join Red L₁ L₂ :=
  join_of_single reflexive_reflTransGen h.to_red


@[to_additive FreeAddGroup.eqvGen_step_iff_join_red]
theorem eqvGen_step_iff_join_red : EqvGen Red.Step L₁ L₂ ↔ Join Red L₁ L₂ :=
  Iff.intro
    (fun h =>
      have : EqvGen (Join Red) L₁ L₂ := h.mono fun _ _ => join_red_of_step
      equivalence_join_red.eqvGen_iff.1 this)
    (join_of_equivalence (Relation.EqvGen.is_equivalence _) fun _ _ =>
      reflTransGen_of_equivalence (Relation.EqvGen.is_equivalence _) EqvGen.rel)


/-- The free group over a type, i.e. the words formed by the elements of the type and their formal
inverses, quotient by one step reduction. -/
@[to_additive "The free additive group over a type, i.e. the words formed by the elements of the
  type and their formal inverses, quotient by one step reduction."]
def FreeGroup (α : Type u) : Type u :=
  Quot <| @FreeGroup.Red.Step α


/-- The canonical map from `List (α × Bool)` to the free group on `α`. -/
@[to_additive "The canonical map from `list (α × bool)` to the free additive group on `α`."]
def mk (L : List (α × Bool)) : FreeGroup α :=
  Quot.mk Red.Step L


@[to_additive (attr := simp)]
theorem quot_mk_eq_mk : Quot.mk Red.Step L = mk L :=
  rfl


@[to_additive (attr := simp)]
theorem quot_lift_mk (β : Type v) (f : List (α × Bool) → β)
    (H : ∀ L₁ L₂, Red.Step L₁ L₂ → f L₁ = f L₂) : Quot.lift f H (mk L) = f L :=
  rfl


@[to_additive (attr := simp)]
theorem quot_liftOn_mk (β : Type v) (f : List (α × Bool) → β)
    (H : ∀ L₁ L₂, Red.Step L₁ L₂ → f L₁ = f L₂) : Quot.liftOn (mk L) f H = f L :=
  rfl


@[to_additive (attr := simp)]
theorem quot_map_mk (β : Type v) (f : List (α × Bool) → List (β × Bool))
    (H : (Red.Step ⇒ Red.Step) f f) : Quot.map f H (mk L) = mk (f L) :=
  rfl


@[to_additive]
instance : One (FreeGroup α) :=
  ⟨mk []⟩


@[to_additive]
theorem one_eq_mk : (1 : FreeGroup α) = mk [] :=
  rfl


@[to_additive]
instance : Inhabited (FreeGroup α) :=
  ⟨1⟩


@[to_additive]
                                                  /-
                                                    α : Type u
                                                    L L₁ L₂ L₃ L₄ : List (Prod α Bool)
                                                    inst✝ : IsEmpty α
                                                    ⊢ Unique (FreeGroup α)
                                                  -/
instance [IsEmpty α] : Unique (FreeGroup α) := by unfold FreeGroup; infer_instance
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


@[to_additive]
instance : Mul (FreeGroup α) :=
  ⟨fun x y =>
    Quot.liftOn x
      (fun L₁ =>
        Quot.liftOn y (fun L₂ => mk <| L₁ ++ L₂) fun _L₂ _L₃ H =>
          Quot.sound <| Red.Step.append_left H)
      fun _L₁ _L₂ H => Quot.inductionOn y fun _L₃ => Quot.sound <| Red.Step.append_right H⟩


@[to_additive (attr := simp)]
theorem mul_mk : mk L₁ * mk L₂ = mk (L₁ ++ L₂) :=
  rfl


/-- Transform a word representing a free group element into a word representing its inverse. -/
@[to_additive "Transform a word representing a free group element into a word representing its
  negative."]
def invRev (w : List (α × Bool)) : List (α × Bool) :=
  (List.map (fun g : α × Bool => (g.1, not g.2)) w).reverse


@[to_additive (attr := simp)]
                                                             /-
                                                               α : Type u
                                                               L₁ : List (Prod α Bool)
                                                               ⊢ Eq (FreeGroup.invRev L₁).length L₁.length
                                                             -/
theorem invRev_length : (invRev L₁).length = L₁.length := by simp [invRev]
                                                             /-
                                                               🎉 no goals
                                                             -/


@[to_additive (attr := simp)]
theorem invRev_invRev : invRev (invRev L₁) = L₁ := by
  /-
    α : Type u
    L₁ : List (Prod α Bool)
    ⊢ Eq (FreeGroup.invRev (FreeGroup.invRev L₁)) L₁
  -/
  simp [invRev, List.map_reverse, Function.comp_def]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
theorem invRev_empty : invRev ([] : List (α × Bool)) = [] :=
  rfl


@[to_additive]
theorem invRev_involutive : Function.Involutive (@invRev α) := fun _ => invRev_invRev


@[to_additive]
theorem invRev_injective : Function.Injective (@invRev α) :=
  invRev_involutive.injective


@[to_additive]
theorem invRev_surjective : Function.Surjective (@invRev α) :=
  invRev_involutive.surjective


@[to_additive]
theorem invRev_bijective : Function.Bijective (@invRev α) :=
  invRev_involutive.bijective


@[to_additive]
instance : Inv (FreeGroup α) :=
  ⟨Quot.map invRev
      (by
        /-
          α : Type u
          L L₁ L₂ L₃ L₄ : List (Prod α Bool)
          ⊢ ∀ ⦃a b : List (Prod α Bool)⦄, FreeGroup.Red.Step a b → FreeGroup.Red.Step (F …
        -/
        intro a b h
        /-
          α : Type u
          L L₁ L₂ L₃ L₄ a b : List (Prod α Bool)
          h : FreeGroup.Red.Step a b
          ⊢ FreeGroup.Red.Step (FreeGroup.invRev a) (FreeGroup.invRev b)
        -/
        cases h
        /-
          case not
          α : Type u
          L L₁ L₂ L₃ L₄ L₁✝ L₂✝ : List (Prod α Bool)
          x✝ : α
          b✝ : Bool
          ⊢ FreeGroup.Red.Step (FreeGroup.invRev (HAppend.hAppend L₁✝ (List.cons { fst : …
        -/
        simp [invRev])⟩
        /-
          🎉 no goals
        -/


@[to_additive (attr := simp)]
theorem inv_mk : (mk L)⁻¹ = mk (invRev L) :=
  rfl


@[to_additive]
theorem Red.Step.invRev {L₁ L₂ : List (α × Bool)} (h : Red.Step L₁ L₂) :
    Red.Step (FreeGroup.invRev L₁) (FreeGroup.invRev L₂) := by
  /-
    α : Type u
    L₁ L₂ : List (Prod α Bool)
    h : FreeGroup.Red.Step L₁ L₂
    ⊢ FreeGroup.Red.Step (FreeGroup.invRev L₁) (FreeGroup.invRev L₂)
  -/
  cases' h with a b x y
  /-
    case not
    α : Type u
    a b : List (Prod α Bool)
    x : α
    y : Bool
    ⊢ FreeGroup.Red.Step (FreeGroup.invRev (HAppend.hAppend a (List.cons { fst :=  …
  -/
  simp [FreeGroup.invRev]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem Red.invRev {L₁ L₂ : List (α × Bool)} (h : Red L₁ L₂) : Red (invRev L₁) (invRev L₂) :=
  Relation.ReflTransGen.lift _ (fun _a _b => Red.Step.invRev) h


@[to_additive (attr := simp)]
theorem Red.step_invRev_iff :
    Red.Step (FreeGroup.invRev L₁) (FreeGroup.invRev L₂) ↔ Red.Step L₁ L₂ :=
               /-
                 α : Type u
                 L₁ L₂ : List (Prod α Bool)
                 h : FreeGroup.Red.Step (FreeGroup.invRev L₁) (FreeGroup.invRev L₂)
                 ⊢ FreeGroup.Red.Step L₁ L₂
               -/
  ⟨fun h => by simpa only [invRev_invRev] using h.invRev, fun h => h.invRev⟩
               /-
                 🎉 no goals
               -/


@[to_additive (attr := simp)]
theorem red_invRev_iff : Red (invRev L₁) (invRev L₂) ↔ Red L₁ L₂ :=
               /-
                 α : Type u
                 L₁ L₂ : List (Prod α Bool)
                 h : FreeGroup.Red (FreeGroup.invRev L₁) (FreeGroup.invRev L₂)
                 ⊢ FreeGroup.Red L₁ L₂
               -/
  ⟨fun h => by simpa only [invRev_invRev] using h.invRev, fun h => h.invRev⟩
               /-
                 🎉 no goals
               -/


@[to_additive]
instance : Group (FreeGroup α) where
  mul := (· * ·)
  one := 1
  inv := Inv.inv
                  /-
                    α : Type u
                    L L₁ L₂ L₃ L₄ : List (Prod α Bool)
                    ⊢ ∀ (a b c : FreeGroup α), Eq (HMul.hMul (HMul.hMul a b) c) (HMul.hMul a (HMul …
                  -/
  mul_assoc := by rintro ⟨L₁⟩ ⟨L₂⟩ ⟨L₃⟩; simp
                                         /-
                                           🎉 no goals
                                         -/
                /-
                  α : Type u
                  L L₁ L₂ L₃ L₄ : List (Prod α Bool)
                  ⊢ ∀ (a : FreeGroup α), Eq (HMul.hMul 1 a) a
                -/
  one_mul := by rintro ⟨L⟩; rfl
                            /-
                              🎉 no goals
                            -/
                /-
                  α : Type u
                  L L₁ L₂ L₃ L₄ : List (Prod α Bool)
                  ⊢ ∀ (a : FreeGroup α), Eq (HMul.hMul a 1) a
                -/
  mul_one := by rintro ⟨L⟩; simp [one_eq_mk]
                            /-
                              🎉 no goals
                            -/
  inv_mul_cancel := by
    /-
      α : Type u
      L L₁ L₂ L₃ L₄ : List (Prod α Bool)
      ⊢ ∀ (a : FreeGroup α), Eq (HMul.hMul (Inv.inv a) a) 1
    -/
    rintro ⟨L⟩
    exact
      List.recOn L rfl fun ⟨x, b⟩ tl ih =>
          Eq.trans (Quot.sound <| by simp [invRev, one_eq_mk]) ih


@[to_additive (attr := simp)]
theorem pow_mk (n : ℕ) : mk L ^ n = mk (List.flatten <| List.replicate n L) :=
  match n with
  | 0 => rfl
                /-
                  α : Type u
                  L : List (Prod α Bool)
                  n✝ n : Nat
                  ⊢ Eq (HPow.hPow (FreeGroup.mk L) (HAdd.hAdd n 1)) (FreeGroup.mk (List.replicat …
                -/
  | n + 1 => by rw [pow_succ', pow_mk, mul_mk, List.replicate_succ, List.flatten_cons]
                /-
                  🎉 no goals
                -/


/-- `of` is the canonical injection from the type to the free group over that type by sending each
element to the equivalence class of the letter that is the element. -/
@[to_additive "`of` is the canonical injection from the type to the free group over that type
  by sending each element to the equivalence class of the letter that is the element."]
def of (x : α) : FreeGroup α :=
  mk [(x, true)]


@[to_additive]
theorem Red.exact : mk L₁ = mk L₂ ↔ Join Red L₁ L₂ :=
  calc
    mk L₁ = mk L₂ ↔ EqvGen Red.Step L₁ L₂ := Iff.intro Quot.eqvGen_exact Quot.eqvGen_sound
    _ ↔ Join Red L₁ L₂ := eqvGen_step_iff_join_red


/-- The canonical map from the type to the free group is an injection. -/
@[to_additive "The canonical map from the type to the additive free group is an injection."]
theorem of_injective : Function.Injective (@of α) := fun _ _ H => by
  /-
    α : Type u
    x✝¹ x✝ : α
    H : Eq (FreeGroup.of x✝¹) (FreeGroup.of x✝)
    ⊢ Eq x✝¹ x✝
  -/
  let ⟨L₁, hx, hy⟩ := Red.exact.1 H
  /-
    α : Type u
    x✝¹ x✝ : α
    H : Eq (FreeGroup.of x✝¹) (FreeGroup.of x✝)
    L₁ : List (Prod α Bool)
    hx : FreeGroup.Red (List.cons { fst := x✝¹, snd := Bool.true } List.nil) L₁
    hy : FreeGroup.Red (List.cons { fst := x✝, snd := Bool.true } List.nil) L₁
    ⊢ Eq x✝¹ x✝
  -/
  simp [Red.singleton_iff] at hx hy; aesop
                                     /-
                                       🎉 no goals
                                     -/


/-- Given `f : α → β` with `β` a group, the canonical map `List (α × Bool) → β` -/
@[to_additive "Given `f : α → β` with `β` an additive group, the canonical map
  `list (α × bool) → β`"]
def Lift.aux : List (α × Bool) → β := fun L =>
  List.prod <| L.map fun x => cond x.2 (f x.1) (f x.1)⁻¹


@[to_additive]
theorem Red.Step.lift {f : α → β} (H : Red.Step L₁ L₂) : Lift.aux f L₁ = Lift.aux f L₂ := by
  /-
    α : Type u
    L₁ L₂ : List (Prod α Bool)
    β : Type v
    inst✝ : Group β
    f : α → β
    H : FreeGroup.Red.Step L₁ L₂
    ⊢ Eq (FreeGroup.Lift.aux f L₁) (FreeGroup.Lift.aux f L₂)
  -/
                                     /-
                                       🎉 no goals
                                     -/
  cases' H with _ _ _ b; cases b <;> simp [Lift.aux]
                                     /-
                                       🎉 no goals
                                     -/


/-- If `β` is a group, then any function from `α` to `β` extends uniquely to a group homomorphism
from the free group over `α` to `β` -/
@[to_additive (attr := simps symm_apply)
  "If `β` is an additive group, then any function from `α` to `β` extends uniquely to an
  additive group homomorphism from the free additive group over `α` to `β`"]
def lift : (α → β) ≃ (FreeGroup α →* β) where
  toFun f :=
    MonoidHom.mk' (Quot.lift (Lift.aux f) fun _ _ => Red.Step.lift) <| by
      /-
        α : Type u
        L L₁ L₂ L₃ L₄ : List (Prod α Bool)
        β : Type v
        inst✝ : Group β
        f✝ : α → β
        x y : FreeGroup α
        f : α → β
        ⊢ ∀ (a b : FreeGroup α), Eq (Quot.lift (FreeGroup.Lift.aux f) ⋯ (HMul.hMul a b …
      -/
      rintro ⟨L₁⟩ ⟨L₂⟩; simp [Lift.aux]
                        /-
                          🎉 no goals
                        -/
  invFun g := g ∘ of
  left_inv f := List.prod_singleton
  right_inv g :=
    MonoidHom.ext <| by
      /-
        α : Type u
        L L₁ L₂ L₃ L₄ : List (Prod α Bool)
        β : Type v
        inst✝ : Group β
        f : α → β
        x y : FreeGroup α
        g : MonoidHom (FreeGroup α) β
        ⊢ ∀ (x : FreeGroup α), Eq (((fun f => MonoidHom.mk' (Quot.lift (FreeGroup.Lift …
      -/
      rintro ⟨L⟩
      exact List.recOn L
        (g.map_one.symm)
        (by
        rintro ⟨x, _ | _⟩ t (ih : _ = g (mk t))
        · show _ = g ((of x)⁻¹ * mk t)
          simpa [Lift.aux] using ih
        · show _ = g (of x * mk t)
          simpa [Lift.aux] using ih)


@[to_additive (attr := simp)]
theorem lift.mk : lift f (mk L) = List.prod (L.map fun x => cond x.2 (f x.1) (f x.1)⁻¹) :=
  rfl


@[to_additive (attr := simp)]
theorem lift.of {x} : lift f (of x) = f x :=
  List.prod_singleton


@[to_additive]
theorem lift.unique (g : FreeGroup α →* β) (hg : ∀ x, g (FreeGroup.of x) = f x) {x} :
    g x = FreeGroup.lift f x :=
  DFunLike.congr_fun (lift.symm_apply_eq.mp (funext hg : g ∘ FreeGroup.of = f)) x


/-- Two homomorphisms out of a free group are equal if they are equal on generators.

See note [partially-applied ext lemmas]. -/
@[to_additive (attr := ext) "Two homomorphisms out of a free additive group are equal if they are
  equal on generators. See note [partially-applied ext lemmas]."]
theorem ext_hom {G : Type*} [Group G] (f g : FreeGroup α →* G) (h : ∀ a, f (of a) = g (of a)) :
    f = g :=
  lift.symm.injective <| funext h


@[to_additive]
theorem lift_of_eq_id (α) : lift of = MonoidHom.id (FreeGroup α) :=
  lift.apply_symm_apply (MonoidHom.id _)


@[to_additive]
theorem lift.of_eq (x : FreeGroup α) : lift FreeGroup.of x = x :=
  DFunLike.congr_fun (lift_of_eq_id α) x


@[to_additive]
theorem lift.range_le {s : Subgroup β} (H : Set.range f ⊆ s) : (lift f).range ≤ s := by
  /-
    α : Type u
    β : Type v
    inst✝ : Group β
    f : α → β
    s : Subgroup β
    H : HasSubset.Subset (Set.range f) ↑s
    ⊢ LE.le (FreeGroup.lift f).range s
  -/
  rintro _ ⟨⟨L⟩, rfl⟩;
  exact List.recOn L s.one_mem fun ⟨x, b⟩ tl ih ↦
    Bool.recOn b (by simpa using s.mul_mem (s.inv_mem <| H ⟨x, rfl⟩) ih)
      (by simpa using s.mul_mem (H ⟨x, rfl⟩) ih)


@[to_additive]
theorem lift.range_eq_closure : (lift f).range = Subgroup.closure (Set.range f) := by
  /-
    α : Type u
    β : Type v
    inst✝ : Group β
    f : α → β
    ⊢ Eq (FreeGroup.lift f).range (Subgroup.closure (Set.range f))
  -/
  apply le_antisymm (lift.range_le Subgroup.subset_closure)
  /-
    α : Type u
    β : Type v
    inst✝ : Group β
    f : α → β
    ⊢ LE.le (Subgroup.closure (Set.range f)) (FreeGroup.lift f).range
  -/
  rw [Subgroup.closure_le]
  /-
    α : Type u
    β : Type v
    inst✝ : Group β
    f : α → β
    ⊢ HasSubset.Subset (Set.range f) ↑(FreeGroup.lift f).range
  -/
  rintro _ ⟨a, rfl⟩
  /-
    case intro
    α : Type u
    β : Type v
    inst✝ : Group β
    f : α → β
    a : α
    ⊢ Membership.mem (↑(FreeGroup.lift f).range) (f a)
  -/
  exact ⟨FreeGroup.of a, by simp only [lift.of]⟩
  /-
    🎉 no goals
  -/


/-- The generators of `FreeGroup α` generate `FreeGroup α`. That is, the subgroup closure of the
set of generators equals `⊤`. -/
@[to_additive (attr := simp)]
theorem closure_range_of (α) :
    Subgroup.closure (Set.range (FreeGroup.of : α → FreeGroup α)) = ⊤ := by
  /-
    α : Type u_1
    ⊢ Eq (Subgroup.closure (Set.range FreeGroup.of)) Top.top
  -/
  rw [← lift.range_eq_closure, lift_of_eq_id]
  /-
    α : Type u_1
    ⊢ Eq (MonoidHom.id (FreeGroup α)).range Top.top
  -/
  exact MonoidHom.range_eq_top.2 Function.surjective_id
  /-
    🎉 no goals
  -/


/-- Any function from `α` to `β` extends uniquely to a group homomorphism from the free group over
  `α` to the free group over `β`. -/
@[to_additive "Any function from `α` to `β` extends uniquely to an additive group homomorphism from
  the additive free group over `α` to the additive free group over `β`."]
def map : FreeGroup α →* FreeGroup β :=
  MonoidHom.mk'
                                                                 /-
                                                                   α : Type u
                                                                   L L₁✝ L₂✝ L₃ L₄ : List (Prod α Bool)
                                                                   β : Type v
                                                                   f : α → β
                                                                   x y : FreeGroup α
                                                                   L₁ L₂ : List (Prod α Bool)
                                                                   H : FreeGroup.Red.Step L₁ L₂
                                                                   ⊢ FreeGroup.Red.Step (List.map (fun x => { fst := f x.1, snd := x.2 }) L₁) (Li …
                                                                 -/
    (Quot.map (List.map fun x => (f x.1, x.2)) fun L₁ L₂ H => by cases H; simp)
                                                                          /-
                                                                            🎉 no goals
                                                                          -/
        /-
          α : Type u
          L L₁ L₂ L₃ L₄ : List (Prod α Bool)
          β : Type v
          f : α → β
          x y : FreeGroup α
          ⊢ ∀ (a b : FreeGroup α), Eq (Quot.map (List.map fun x => { fst := f x.1, snd : …
        -/
    (by rintro ⟨L₁⟩ ⟨L₂⟩; simp)
                          /-
                            🎉 no goals
                          -/


@[to_additive (attr := simp)]
theorem map.mk : map f (mk L) = mk (L.map fun x => (f x.1, x.2)) :=
  rfl


@[to_additive (attr := simp)]
                                                      /-
                                                        α : Type u
                                                        x : FreeGroup α
                                                        ⊢ Eq ((FreeGroup.map _root_.id) x) x
                                                      -/
theorem map.id (x : FreeGroup α) : map id x = x := by rcases x with ⟨L⟩; simp [List.map_id']
                                                                         /-
                                                                           🎉 no goals
                                                                         -/


@[to_additive (attr := simp)]
theorem map.id' (x : FreeGroup α) : map (fun z => z) x = x :=
  map.id x


@[to_additive]
theorem map.comp {γ : Type w} (f : α → β) (g : β → γ) (x) :
    map g (map f x) = map (g ∘ f) x := by
  /-
    α : Type u
    β : Type v
    γ : Type w
    f : α → β
    g : β → γ
    x : FreeGroup α
    ⊢ Eq ((FreeGroup.map g) ((FreeGroup.map f) x)) ((FreeGroup.map (Function.comp  …
  -/
  rcases x with ⟨L⟩; simp [Function.comp_def]
                     /-
                       🎉 no goals
                     -/


@[to_additive (attr := simp)]
theorem map.of {x} : map f (of x) = of (f x) :=
  rfl


@[to_additive]
theorem map.unique (g : FreeGroup α →* FreeGroup β)
    (hg : ∀ x, g (FreeGroup.of x) = FreeGroup.of (f x)) :
    ∀ {x}, g x = map f x := by
  /-
    α : Type u
    β : Type v
    f : α → β
    g : MonoidHom (FreeGroup α) (FreeGroup β)
    hg : ∀ (x : α), Eq (g (FreeGroup.of x)) (FreeGroup.of (f x))
    ⊢ ∀ {x : FreeGroup α}, Eq (g x) ((FreeGroup.map f) x)
  -/
  rintro ⟨L⟩
  exact List.recOn L g.map_one fun ⟨x, b⟩ t (ih : g (FreeGroup.mk t) = map f (FreeGroup.mk t)) =>
    Bool.recOn b
      (show g ((FreeGroup.of x)⁻¹ * FreeGroup.mk t) =
          FreeGroup.map f ((FreeGroup.of x)⁻¹ * FreeGroup.mk t) by
        simp [g.map_mul, g.map_inv, hg, ih])
      (show g (FreeGroup.of x * FreeGroup.mk t) =
          FreeGroup.map f (FreeGroup.of x * FreeGroup.mk t) by simp [g.map_mul, hg, ih])


@[to_additive]
theorem map_eq_lift : map f x = lift (of ∘ f) x :=
                                      /-
                                        α : Type u
                                        β : Type v
                                        f : α → β
                                        x✝ : FreeGroup α
                                        x : α
                                        ⊢ Eq ((FreeGroup.lift (Function.comp FreeGroup.of f)) (FreeGroup.of x)) (FreeG …
                                      -/
  Eq.symm <| map.unique _ fun x => by simp
                                      /-
                                        🎉 no goals
                                      -/


/-- Equivalent types give rise to multiplicatively equivalent free groups.

The converse can be found in `GroupTheory.FreeAbelianGroupFinsupp`,
as `Equiv.of_freeGroupEquiv`
 -/
@[to_additive (attr := simps apply)
  "Equivalent types give rise to additively equivalent additive free groups."]
def freeGroupCongr {α β} (e : α ≃ β) : FreeGroup α ≃* FreeGroup β where
  toFun := map e
  invFun := map e.symm
                   /-
                     α✝ : Type u
                     L L₁ L₂ L₃ L₄ : List (Prod α✝ Bool)
                     β✝ : Type v
                     f : α✝ → β✝
                     x✝ y : FreeGroup α✝
                     α : Type ?u.66025
                     β : Type ?u.66026
                     e : Equiv α β
                     x : FreeGroup α
                     ⊢ Eq ((FreeGroup.map ⇑e.symm) ((FreeGroup.map ⇑e) x)) x
                   -/
  left_inv x := by simp [Function.comp, map.comp]
                   /-
                     🎉 no goals
                   -/
                    /-
                      α✝ : Type u
                      L L₁ L₂ L₃ L₄ : List (Prod α✝ Bool)
                      β✝ : Type v
                      f : α✝ → β✝
                      x✝ y : FreeGroup α✝
                      α : Type ?u.66025
                      β : Type ?u.66026
                      e : Equiv α β
                      x : FreeGroup β
                      ⊢ Eq ((FreeGroup.map ⇑e) ((FreeGroup.map ⇑e.symm) x)) x
                    -/
  right_inv x := by simp [Function.comp, map.comp]
                    /-
                      🎉 no goals
                    -/
  map_mul' := MonoidHom.map_mul _


@[to_additive (attr := simp)]
theorem freeGroupCongr_refl : freeGroupCongr (Equiv.refl α) = MulEquiv.refl _ :=
  MulEquiv.ext map.id


@[to_additive (attr := simp)]
theorem freeGroupCongr_symm {α β} (e : α ≃ β) : (freeGroupCongr e).symm = freeGroupCongr e.symm :=
  rfl


@[to_additive]
theorem freeGroupCongr_trans {α β γ} (e : α ≃ β) (f : β ≃ γ) :
    (freeGroupCongr e).trans (freeGroupCongr f) = freeGroupCongr (e.trans f) :=
  MulEquiv.ext <| map.comp _ _


/-- If `α` is a group, then any function from `α` to `α` extends uniquely to a homomorphism from the
free group over `α` to `α`. This is the multiplicative version of `FreeGroup.sum`. -/
@[to_additive "If `α` is an additive group, then any function from `α` to `α` extends uniquely to an
  additive homomorphism from the additive free group over `α` to `α`."]
def prod : FreeGroup α →* α :=
  lift id


@[to_additive (attr := simp)]
theorem prod_mk : prod (mk L) = List.prod (L.map fun x => cond x.2 x.1 x.1⁻¹) :=
  rfl


@[to_additive (attr := simp)]
theorem prod.of {x : α} : prod (of x) = x :=
  lift.of


@[to_additive]
theorem prod.unique (g : FreeGroup α →* α) (hg : ∀ x, g (FreeGroup.of x) = x) {x} : g x = prod x :=
  lift.unique g hg


@[to_additive]
theorem lift_eq_prod_map {β : Type v} [Group β] {f : α → β} {x} : lift f x = prod (map f x) := by
  /-
    α : Type u
    β : Type v
    inst✝ : Group β
    f : α → β
    x : FreeGroup α
    ⊢ Eq ((FreeGroup.lift f) x) (FreeGroup.prod ((FreeGroup.map f) x))
  -/
  rw [← lift.unique (prod.comp (map f))]
    /-
      α : Type u
      β : Type v
      inst✝ : Group β
      f : α → β
      x : FreeGroup α
      ⊢ Eq ((FreeGroup.prod.comp (FreeGroup.map f)) x) (FreeGroup.prod ((FreeGroup.m …
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case hg
      α : Type u
      β : Type v
      inst✝ : Group β
      f : α → β
      x : FreeGroup α
      ⊢ ∀ (x : α), Eq ((FreeGroup.prod.comp (FreeGroup.map f)) (FreeGroup.of x)) (f x)
    -/
  · simp
    /-
      🎉 no goals
    -/


/-- If `α` is a group, then any function from `α` to `α` extends uniquely to a homomorphism from the
free group over `α` to `α`. This is the additive version of `Prod`. -/
def sum : α :=
  @prod (Multiplicative _) _ x


@[simp]
theorem sum_mk : sum (mk L) = List.sum (L.map fun x => cond x.2 x.1 (-x.1)) :=
  rfl


@[simp]
theorem sum.of {x : α} : sum (of x) = x :=
  @prod.of _ (_) _

-- note: there are no bundled homs with different notation in the domain and codomain, so we copy
-- these manually

@[simp]
theorem sum.map_mul : sum (x * y) = sum x + sum y :=
  (@prod (Multiplicative _) _).map_mul _ _


@[simp]
theorem sum.map_one : sum (1 : FreeGroup α) = 0 :=
  (@prod (Multiplicative _) _).map_one


@[simp]
theorem sum.map_inv : sum x⁻¹ = -sum x :=
  (prod : FreeGroup (Multiplicative α) →* Multiplicative α).map_inv _


/-- The bijection between the free group on the empty type, and a type with one element. -/
@[to_additive "The bijection between the additive free group on the empty type, and a type with one
  element."]
def freeGroupEmptyEquivUnit : FreeGroup Empty ≃ Unit where
  toFun _ := ()
  invFun _ := 1
                 /-
                   α : Type u
                   L L₁ L₂ L₃ L₄ : List (Prod α Bool)
                   ⊢ Function.LeftInverse (fun x => 1) fun x => Unit.unit
                 -/
  left_inv := by rintro ⟨_ | ⟨⟨⟨⟩, _⟩, _⟩⟩; rfl
                                            /-
                                              🎉 no goals
                                            -/
  right_inv := fun ⟨⟩ => rfl


/-- The bijection between the free group on a singleton, and the integers. -/
def freeGroupUnitEquivInt : FreeGroup Unit ≃ ℤ where
  toFun x := sum (by
    /-
      α : Type u
      L L₁ L₂ L₃ L₄ : List (Prod α Bool)
      x : FreeGroup Unit
      ⊢ FreeGroup Int
    -/
    revert x
    /-
      α : Type u
      L L₁ L₂ L₃ L₄ : List (Prod α Bool)
      ⊢ FreeGroup Unit → FreeGroup Int
    -/
    exact ↑(map fun _ => (1 : ℤ)))
    /-
      🎉 no goals
    -/
  invFun x := of () ^ x
  left_inv := by
    /-
      α : Type u
      L L₁ L₂ L₃ L₄ : List (Prod α Bool)
      ⊢ Function.LeftInverse (fun x => HPow.hPow (FreeGroup.of Unit.unit) x) fun x = …
    -/
    rintro ⟨L⟩
    /-
      case mk
      α : Type u
      L✝ L₁ L₂ L₃ L₄ : List (Prod α Bool)
      x✝ : FreeGroup Unit
      L : List (Prod Unit Bool)
      ⊢ Eq ((fun x => HPow.hPow (FreeGroup.of Unit.unit) x) ((fun x => ((FreeGroup.m …
    -/
    simp only [quot_mk_eq_mk, map.mk, sum_mk, List.map_map]
    exact List.recOn L
     (by rfl)
     (fun ⟨⟨⟩, b⟩ tl ih => by
        cases b <;> simp [zpow_add] at ih ⊢ <;> rw [ih] <;> rfl)
  right_inv x :=
                           /-
                             α : Type u
                             L L₁ L₂ L₃ L₄ : List (Prod α Bool)
                             x : Int
                             ⊢ Eq ((fun x => ((FreeGroup.map fun x => 1) x).sum) ((fun x => HPow.hPow (Free …
                           -/
    Int.induction_on x (by simp)
                           /-
                             🎉 no goals
                           -/
      (fun i ih => by
        /-
          α : Type u
          L L₁ L₂ L₃ L₄ : List (Prod α Bool)
          x : Int
          i : Nat
          ih : Eq ((fun x => ((FreeGroup.map fun x => 1) x).sum) ((fun x => HPow.hPow (F …
          ⊢ Eq ((fun x => ((FreeGroup.map fun x => 1) x).sum) ((fun x => HPow.hPow (Free …
        -/
        simp only [zpow_natCast, map_pow, map.of] at ih
        /-
          α : Type u
          L L₁ L₂ L₃ L₄ : List (Prod α Bool)
          x : Int
          i : Nat
          ih : Eq (HPow.hPow (FreeGroup.of 1) i).sum ↑i
          ⊢ Eq ((fun x => ((FreeGroup.map fun x => 1) x).sum) ((fun x => HPow.hPow (Free …
        -/
        simp [zpow_add, ih])
        /-
          🎉 no goals
        -/
      (fun i ih => by
        /-
          α : Type u
          L L₁ L₂ L₃ L₄ : List (Prod α Bool)
          x : Int
          i : Nat
          ih : Eq ((fun x => ((FreeGroup.map fun x => 1) x).sum) ((fun x => HPow.hPow (F …
          ⊢ Eq ((fun x => ((FreeGroup.map fun x => 1) x).sum) ((fun x => HPow.hPow (Free …
        -/
        simp only [zpow_neg, zpow_natCast, map_inv, map_pow, map.of, sum.map_inv, neg_inj] at ih
        /-
          α : Type u
          L L₁ L₂ L₃ L₄ : List (Prod α Bool)
          x : Int
          i : Nat
          ih : Eq (HPow.hPow (FreeGroup.of 1) i).sum ↑i
          ⊢ Eq ((fun x => ((FreeGroup.map fun x => 1) x).sum) ((fun x => HPow.hPow (Free …
        -/
        simp [zpow_add, ih, sub_eq_add_neg])
        /-
          🎉 no goals
        -/


@[to_additive]
instance : Monad FreeGroup.{u} where
  pure {_α} := of
  map {_α} {_β} {f} := map f
  bind {_α} {_β} {x} {f} := lift f x


@[to_additive (attr := elab_as_elim, induction_eliminator)]
protected theorem induction_on {C : FreeGroup α → Prop} (z : FreeGroup α) (C1 : C 1)
    (Cp : ∀ x, C <| pure x) (Ci : ∀ x, C (pure x) → C (pure x)⁻¹)
    (Cm : ∀ x y, C x → C y → C (x * y)) : C z :=
  Quot.inductionOn z fun L =>
    List.recOn L C1 fun ⟨x, b⟩ _tl ih => Bool.recOn b (Cm _ _ (Ci _ <| Cp x) ih) (Cm _ _ (Cp x) ih)


@[to_additive]
theorem map_pure (f : α → β) (x : α) : f <$> (pure x : FreeGroup α) = pure (f x) :=
  map.of


@[to_additive (attr := simp)]
theorem map_one (f : α → β) : f <$> (1 : FreeGroup α) = 1 :=
  (map f).map_one


@[to_additive (attr := simp)]
theorem map_mul (f : α → β) (x y : FreeGroup α) : f <$> (x * y) = f <$> x * f <$> y :=
  (map f).map_mul x y


@[to_additive (attr := simp)]
theorem map_inv (f : α → β) (x : FreeGroup α) : f <$> x⁻¹ = (f <$> x)⁻¹ :=
  (map f).map_inv x


@[to_additive]
theorem pure_bind (f : α → FreeGroup β) (x) : pure x >>= f = f x :=
  lift.of


@[to_additive (attr := simp)]
theorem one_bind (f : α → FreeGroup β) : 1 >>= f = 1 :=
  (lift f).map_one


@[to_additive (attr := simp)]
theorem mul_bind (f : α → FreeGroup β) (x y : FreeGroup α) : x * y >>= f = (x >>= f) * (y >>= f) :=
  (lift f).map_mul _ _


@[to_additive (attr := simp)]
theorem inv_bind (f : α → FreeGroup β) (x : FreeGroup α) : x⁻¹ >>= f = (x >>= f)⁻¹ :=
  (lift f).map_inv _


@[to_additive]
                                        /-
                                          α : Type u
                                          L L₁ L₂ L₃ L₄ : List (Prod α Bool)
                                          β : Type u
                                          ⊢ ∀ {α β : Type u} (x : α) (y : FreeGroup β), Eq (Functor.mapConst x y) (Funct …
                                        -/
                                        /-
                                          🎉 no goals
                                        -/
                                                                                   /-
                                                                                     α : Type u
                                                                                     L L₁ L₂ L₃ L₄ : List (Prod α Bool)
                                                                                     β α✝ : Type u
                                                                                     x✝ : FreeGroup α✝
                                                                                     x : α✝
                                                                                     ih : Eq (Functor.map id (Pure.pure x)) (Pure.pure x)
                                                                                     ⊢ Eq (Functor.map id (Inv.inv (Pure.pure x))) (Inv.inv (Pure.pure x))
                                                                                   -/
                                        /-
                                          🎉 no goals
                                        -/
                                                                                   /-
                                                                                     🎉 no goals
                                                                                   -/
                            /-
                              α : Type u
                              L L₁ L₂ L₃ L₄ : List (Prod α Bool)
                              β α✝ : Type u
                              x✝ x y : FreeGroup α✝
                              ihx : Eq (Functor.map id x) x
                              ihy : Eq (Functor.map id y) y
                              ⊢ Eq (Functor.map id (HMul.hMul x y)) (HMul.hMul x y)
                            -/
                                        /-
                                          🎉 no goals
                                        -/
                            /-
                              🎉 no goals
                            -/
instance : LawfulMonad FreeGroup.{u} := LawfulMonad.mk'
                                        /-
                                          🎉 no goals
                                        -/
  (id_map := fun x =>
          /-
            α : Type u
            L L₁ L₂ L₃ L₄ : List (Prod α Bool)
            β α✝ β✝ γ✝ : Type u
            x : FreeGroup α✝
            ⊢ ∀ (f : α✝ → FreeGroup β✝) (g : β✝ → FreeGroup γ✝), Eq (Bind.bind (Bind.bind  …
          -/
    FreeGroup.induction_on x (map_one id) (fun x => map_pure id x) (fun x ih => by rw [map_inv, ih])
                  /-
                    🎉 no goals
                  -/
                   /-
                     α : Type u
                     L L₁ L₂ L₃ L₄ : List (Prod α Bool)
                     β α✝ β✝ γ✝ : Type u
                     x✝ : FreeGroup α✝
                     x : α✝
                     ⊢ ∀ (f : α✝ → FreeGroup β✝) (g : β✝ → FreeGroup γ✝), Eq (Bind.bind (Bind.bind  …
                   -/
      fun x y ihx ihy => by rw [map_mul, ihx, ihy])
                           /-
                             🎉 no goals
                           -/
                      /-
                        α : Type u
                        L L₁ L₂ L₃ L₄ : List (Prod α Bool)
                        β α✝ β✝ γ✝ : Type u
                        x✝ : FreeGroup α✝
                        x : α✝
                        ih : ∀ (f : α✝ → FreeGroup β✝) (g : β✝ → FreeGroup γ✝), Eq (Bind.bind (Bind.bi …
                        ⊢ ∀ (f : α✝ → FreeGroup β✝) (g : β✝ → FreeGroup γ✝), Eq (Bind.bind (Bind.bind  …
                      -/
  (pure_bind := fun x f => pure_bind f x)
                                                         /-
                                                           🎉 no goals
                                                         -/
                             /-
                               α : Type u
                               L L₁ L₂ L₃ L₄ : List (Prod α Bool)
                               β α✝ β✝ γ✝ : Type u
                               x✝ x y : FreeGroup α✝
                               ihx : ∀ (f : α✝ → FreeGroup β✝) (g : β✝ → FreeGroup γ✝), Eq (Bind.bind (Bind.b …
                               ihy : ∀ (f : α✝ → FreeGroup β✝) (g : β✝ → FreeGroup γ✝), Eq (Bind.bind (Bind.b …
                               ⊢ ∀ (f : α✝ → FreeGroup β✝) (g : β✝ → FreeGroup γ✝), Eq (Bind.bind (Bind.bind  …
                             -/
  (bind_assoc := fun x =>
                                                                /-
                                                                  🎉 no goals
                                                                -/
    FreeGroup.induction_on x
      (by intros; iterate 3 rw [one_bind])
      (fun x => by intros; iterate 2 rw [pure_bind])
      (fun x ih => by intros; (iterate 3 rw [inv_bind]); rw [ih])
                                 /-
                                   α : Type u
                                   L L₁ L₂ L₃ L₄ : List (Prod α Bool)
                                   β α✝ β✝ : Type u
                                   f : α✝ → β✝
                                   x : FreeGroup α✝
                                   ⊢ Eq (Bind.bind 1 fun y => Pure.pure (f y)) (Functor.map f 1)
                                 -/
                                 /-
                                   🎉 no goals
                                 -/
      (fun x y ihx ihy => by intros; (iterate 3 rw [mul_bind]); rw [ihx, ihy]))
                                                                      /-
                                                                        🎉 no goals
                                                                      -/
                      /-
                        α : Type u
                        L L₁ L₂ L₃ L₄ : List (Prod α Bool)
                        β α✝ β✝ : Type u
                        f : α✝ → β✝
                        x✝ : FreeGroup α✝
                        x : α✝
                        ih : Eq (Bind.bind (Pure.pure x) fun y => Pure.pure (f y)) (Functor.map f (Pur …
                        ⊢ Eq (Bind.bind (Inv.inv (Pure.pure x)) fun y => Pure.pure (f y)) (Functor.map …
                      -/
  (bind_pure_comp := fun f x =>
                      /-
                        🎉 no goals
                      -/
      /-
        α : Type u
        L L₁ L₂ L₃ L₄ : List (Prod α Bool)
        β α✝ β✝ : Type u
        f : α✝ → β✝
        x✝ x y : FreeGroup α✝
        ihx : Eq (Bind.bind x fun y => Pure.pure (f y)) (Functor.map f x)
        ihy : Eq (Bind.bind y fun y => Pure.pure (f y)) (Functor.map f y)
        ⊢ Eq (Bind.bind (HMul.hMul x y) fun y => Pure.pure (f y)) (Functor.map f (HMul …
      -/
    FreeGroup.induction_on x (by rw [one_bind, map_one]) (fun x => by rw [pure_bind, map_pure])
      /-
        🎉 no goals
      -/
      (fun x ih => by rw [inv_bind, map_inv, ih]) fun x y ihx ihy => by
      rw [mul_bind, map_mul, ihx, ihy])


