/-- `Path a b` is the type of paths from `a` to `b` through the arrows of `G`. -/
inductive Path {V : Type u} [Quiver.{v} V] (a : V) : V → Sort max (u + 1) v
  | nil : Path a a
  | cons : ∀ {b c : V}, Path a b → (b ⟶ c) → Path a c

-- See issue https://github.com/leanprover/lean4/issues/2049

compile_inductive% Path


/-- An arrow viewed as a path of length one. -/
def Hom.toPath {V} [Quiver V] {a b : V} (e : a ⟶ b) : Path a b :=
  Path.nil.cons e


lemma nil_ne_cons (p : Path a b) (e : b ⟶ a) : Path.nil ≠ p.cons e :=
              /-
                V : Type u
                inst✝ : Quiver V
                a b : V
                p : Quiver.Path a b
                e : Quiver.Hom b a
                h : Eq Quiver.Path.nil (p.cons e)
                ⊢ False
              -/
  fun h => by injection h
              /-
                🎉 no goals
              -/


lemma cons_ne_nil (p : Path a b) (e : b ⟶ a) : p.cons e ≠ Path.nil :=
              /-
                V : Type u
                inst✝ : Quiver V
                a b : V
                p : Quiver.Path a b
                e : Quiver.Hom b a
                h : Eq (p.cons e) Quiver.Path.nil
                ⊢ False
              -/
  fun h => by injection h
              /-
                🎉 no goals
              -/


lemma obj_eq_of_cons_eq_cons {p : Path a b} {p' : Path a c}
                                                                       /-
                                                                         V : Type u
                                                                         inst✝ : Quiver V
                                                                         a b c d : V
                                                                         p : Quiver.Path a b
                                                                         p' : Quiver.Path a c
                                                                         e : Quiver.Hom b d
                                                                         e' : Quiver.Hom c d
                                                                         h : Eq (p.cons e) (p'.cons e')
                                                                         ⊢ Eq b c
                                                                       -/
    {e : b ⟶ d} {e' : c ⟶ d} (h : p.cons e = p'.cons e') : b = c := by injection h
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


lemma heq_of_cons_eq_cons {p : Path a b} {p' : Path a c}
                                                                          /-
                                                                            V : Type u
                                                                            inst✝ : Quiver V
                                                                            a b c d : V
                                                                            p : Quiver.Path a b
                                                                            p' : Quiver.Path a c
                                                                            e : Quiver.Hom b d
                                                                            e' : Quiver.Hom c d
                                                                            h : Eq (p.cons e) (p'.cons e')
                                                                            ⊢ HEq p p'
                                                                          -/
    {e : b ⟶ d} {e' : c ⟶ d} (h : p.cons e = p'.cons e') : HEq p p' := by injection h
                                                                          /-
                                                                            🎉 no goals
                                                                          -/


lemma hom_heq_of_cons_eq_cons {p : Path a b} {p' : Path a c}
                                                                          /-
                                                                            V : Type u
                                                                            inst✝ : Quiver V
                                                                            a b c d : V
                                                                            p : Quiver.Path a b
                                                                            p' : Quiver.Path a c
                                                                            e : Quiver.Hom b d
                                                                            e' : Quiver.Hom c d
                                                                            h : Eq (p.cons e) (p'.cons e')
                                                                            ⊢ HEq e e'
                                                                          -/
    {e : b ⟶ d} {e' : c ⟶ d} (h : p.cons e = p'.cons e') : HEq e e' := by injection h
                                                                          /-
                                                                            🎉 no goals
                                                                          -/


/-- The length of a path is the number of arrows it uses. -/
def length {a : V} : ∀ {b : V}, Path a b → ℕ
  | _, nil => 0
  | _, cons p _ => p.length + 1


instance {a : V} : Inhabited (Path a a) :=
  ⟨nil⟩


@[simp]
theorem length_nil {a : V} : (nil : Path a a).length = 0 :=
  rfl


@[simp]
theorem length_cons (a b c : V) (p : Path a b) (e : b ⟶ c) : (p.cons e).length = p.length + 1 :=
  rfl


theorem eq_of_length_zero (p : Path a b) (hzero : p.length = 0) : a = b := by
  /-
    V : Type u
    inst✝ : Quiver V
    a b : V
    p : Quiver.Path a b
    hzero : Eq p.length 0
    ⊢ Eq a b
  -/
  cases p
    /-
      case nil
      V : Type u
      inst✝ : Quiver V
      a : V
      hzero : Eq Quiver.Path.nil.length 0
      ⊢ Eq a a
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case cons
      V : Type u
      inst✝ : Quiver V
      a b b✝ : V
      a✝¹ : Quiver.Path a b✝
      a✝ : Quiver.Hom b✝ b
      hzero : Eq (a✝¹.cons a✝).length 0
      ⊢ Eq a b
    -/
  · cases Nat.succ_ne_zero _ hzero
    /-
      🎉 no goals
    -/


theorem eq_nil_of_length_zero (p : Path a a) (hzero : p.length = 0) : p = nil := by
  /-
    V : Type u
    inst✝ : Quiver V
    a : V
    p : Quiver.Path a a
    hzero : Eq p.length 0
    ⊢ Eq p Quiver.Path.nil
  -/
  cases p
    /-
      case nil
      V : Type u
      inst✝ : Quiver V
      a : V
      hzero : Eq Quiver.Path.nil.length 0
      ⊢ Eq Quiver.Path.nil Quiver.Path.nil
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case cons
      V : Type u
      inst✝ : Quiver V
      a b✝ : V
      a✝¹ : Quiver.Path a b✝
      a✝ : Quiver.Hom b✝ a
      hzero : Eq (a✝¹.cons a✝).length 0
      ⊢ Eq (a✝¹.cons a✝) Quiver.Path.nil
    -/
  · simp at hzero
    /-
      🎉 no goals
    -/


/-- Composition of paths. -/
def comp {a b : V} : ∀ {c}, Path a b → Path b c → Path a c
  | _, p, nil => p
  | _, p, cons q e => (p.comp q).cons e


@[simp]
theorem comp_cons {a b c d : V} (p : Path a b) (q : Path b c) (e : c ⟶ d) :
    p.comp (q.cons e) = (p.comp q).cons e :=
  rfl


@[simp]
theorem comp_nil {a b : V} (p : Path a b) : p.comp Path.nil = p :=
  rfl


@[simp]
theorem nil_comp {a : V} : ∀ {b} (p : Path a b), Path.nil.comp p = p
  | _, nil => rfl
                      /-
                        V : Type u
                        inst✝ : Quiver V
                        a x✝ b✝ : V
                        p : Quiver.Path a b✝
                        a✝ : Quiver.Hom b✝ x✝
                        ⊢ Eq (Quiver.Path.nil.comp (p.cons a✝)) (p.cons a✝)
                      -/
  | _, cons p _ => by rw [comp_cons, nil_comp p]
                      /-
                        🎉 no goals
                      -/


@[simp]
theorem comp_assoc {a b c : V} :
    ∀ {d} (p : Path a b) (q : Path b c) (r : Path c d), (p.comp q).comp r = p.comp (q.comp r)
  | _, _, _, nil => rfl
                            /-
                              V : Type u
                              inst✝ : Quiver V
                              a b c x✝ : V
                              p : Quiver.Path a b
                              q : Quiver.Path b c
                              b✝ : V
                              r : Quiver.Path c b✝
                              a✝ : Quiver.Hom b✝ x✝
                              ⊢ Eq ((p.comp q).comp (r.cons a✝)) (p.comp (q.comp (r.cons a✝)))
                            -/
  | _, p, q, cons r _ => by rw [comp_cons, comp_cons, comp_cons, comp_assoc p q r]
                            /-
                              🎉 no goals
                            -/


@[simp]
theorem length_comp (p : Path a b) : ∀ {c} (q : Path b c), (p.comp q).length = p.length + q.length
  | _, nil => rfl
  | _, cons _ _ => congr_arg Nat.succ (length_comp _ _)


theorem comp_inj {p₁ p₂ : Path a b} {q₁ q₂ : Path b c} (hq : q₁.length = q₂.length) :
    p₁.comp q₁ = p₂.comp q₂ ↔ p₁ = p₂ ∧ q₁ = q₂ := by
  /-
    V : Type u
    inst✝ : Quiver V
    a b c : V
    p₁ p₂ : Quiver.Path a b
    q₁ q₂ : Quiver.Path b c
    hq : Eq q₁.length q₂.length
    ⊢ Iff (Eq (p₁.comp q₁) (p₂.comp q₂)) (And (Eq p₁ p₂) (Eq q₁ q₂))
  -/
  refine ⟨fun h => ?_, by rintro ⟨rfl, rfl⟩; rfl⟩
  induction q₁ with
  | nil =>
    rcases q₂ with _ | ⟨q₂, f₂⟩
    · exact ⟨h, rfl⟩
    · cases hq
  | cons q₁ f₁ ih =>
    rcases q₂ with _ | ⟨q₂, f₂⟩
    · cases hq
    · simp only [comp_cons, cons.injEq] at h
      obtain rfl := h.1
      obtain ⟨rfl, rfl⟩ := ih (Nat.succ.inj hq) h.2.1.eq
      rw [h.2.2.eq]
      exact ⟨rfl, rfl⟩


theorem comp_inj' {p₁ p₂ : Path a b} {q₁ q₂ : Path b c} (h : p₁.length = p₂.length) :
    p₁.comp q₁ = p₂.comp q₂ ↔ p₁ = p₂ ∧ q₁ = q₂ :=
  ⟨fun h_eq => (comp_inj <| Nat.add_left_cancel (n := p₂.length) <|
       /-
         V : Type u
         inst✝ : Quiver V
         a b c : V
         p₁ p₂ : Quiver.Path a b
         q₁ q₂ : Quiver.Path b c
         h : Eq p₁.length p₂.length
         h_eq : Eq (p₁.comp q₁) (p₂.comp q₂)
         ⊢ Eq (HAdd.hAdd p₂.length q₁.length) (HAdd.hAdd p₂.length q₂.length)
       -/
    by simpa [h] using congr_arg length h_eq).1 h_eq,
       /-
         🎉 no goals
       -/
      /-
        V : Type u
        inst✝ : Quiver V
        a b c : V
        p₁ p₂ : Quiver.Path a b
        q₁ q₂ : Quiver.Path b c
        h : Eq p₁.length p₂.length
        ⊢ And (Eq p₁ p₂) (Eq q₁ q₂) → Eq (p₁.comp q₁) (p₂.comp q₂)
      -/
   by rintro ⟨rfl, rfl⟩; rfl⟩
                         /-
                           🎉 no goals
                         -/


theorem comp_injective_left (q : Path b c) : Injective fun p : Path a b => p.comp q :=
  fun _ _ h => ((comp_inj rfl).1 h).1


theorem comp_injective_right (p : Path a b) : Injective (p.comp : Path b c → Path a c) :=
  fun _ _ h => ((comp_inj' rfl).1 h).2


@[simp]
theorem comp_inj_left {p₁ p₂ : Path a b} {q : Path b c} : p₁.comp q = p₂.comp q ↔ p₁ = p₂ :=
  q.comp_injective_left.eq_iff


@[simp]
theorem comp_inj_right {p : Path a b} {q₁ q₂ : Path b c} : p.comp q₁ = p.comp q₂ ↔ q₁ = q₂ :=
  p.comp_injective_right.eq_iff


lemma eq_toPath_comp_of_length_eq_succ (p : Path a b) {n : ℕ}
    (hp : p.length = n + 1) :
    ∃ (c : V) (f : a ⟶ c) (q : Quiver.Path c b) (_ : q.length = n),
      p = f.toPath.comp q := by
  induction p generalizing n with
  | nil => simp at hp
  | @cons c d p q h =>
    cases n
    · rw [length_cons, Nat.zero_add, Nat.add_left_eq_self] at hp
      obtain rfl := eq_of_length_zero p hp
      obtain rfl := eq_nil_of_length_zero p hp
      exact ⟨d, q, nil, rfl, rfl⟩
    · rw [length_cons, Nat.add_right_cancel_iff] at hp
      obtain ⟨x, q'', p'', hl, rfl⟩ := h hp
      exact ⟨x, q'', p''.cons q, by simpa, rfl⟩


/-- Turn a path into a list. The list contains `a` at its head, but not `b` a priori. -/
@[simp]
def toList : ∀ {b : V}, Path a b → List V
  | _, nil => []
  | _, @cons _ _ _ c _ p _ => c :: p.toList


/-- `Quiver.Path.toList` is a contravariant functor. The inversion comes from `Quiver.Path` and
`List` having different preferred directions for adding elements. -/
@[simp]
theorem toList_comp (p : Path a b) : ∀ {c} (q : Path b c), (p.comp q).toList = q.toList ++ p.toList
                 /-
                   V : Type u
                   inst✝ : Quiver V
                   a b : V
                   p : Quiver.Path a b
                   ⊢ Eq (p.comp Quiver.Path.nil).toList (HAppend.hAppend Quiver.Path.nil.toList p …
                 -/
  | _, nil => by simp
                 /-
                   🎉 no goals
                 -/
                                 /-
                                   V : Type u
                                   inst✝ : Quiver V
                                   a b : V
                                   p : Quiver.Path a b
                                   x✝ d : V
                                   q : Quiver.Path b d
                                   a✝ : Quiver.Hom d x✝
                                   ⊢ Eq (p.comp (q.cons a✝)).toList (HAppend.hAppend (q.cons a✝).toList p.toList)
                                 -/
  | _, @cons _ _ _ d _ q _ => by simp [toList_comp]
                                 /-
                                   🎉 no goals
                                 -/


theorem toList_chain_nonempty :
    ∀ {b} (p : Path a b), p.toList.Chain (fun x y => Nonempty (y ⟶ x)) b
  | _, nil => List.Chain.nil
  | _, cons p f => p.toList_chain_nonempty.cons ⟨f⟩


theorem toList_injective (a : V) : ∀ b, Injective (toList : Path a b → List V)
  | _, nil, nil, _ => rfl
                                         /-
                                           V : Type u
                                           inst✝¹ : Quiver V
                                           inst✝ : ∀ (a b : V), Subsingleton (Quiver.Hom a b)
                                           a c : V
                                           p : Quiver.Path a c
                                           f : Quiver.Hom c a
                                           h : Eq Quiver.Path.nil.toList (p.cons f).toList
                                           ⊢ Eq Quiver.Path.nil (p.cons f)
                                         -/
  | _, nil, @cons _ _ _ c _ p f, h => by cases h
                                         /-
                                           🎉 no goals
                                         -/
                                         /-
                                           V : Type u
                                           inst✝¹ : Quiver V
                                           inst✝ : ∀ (a b : V), Subsingleton (Quiver.Hom a b)
                                           a c : V
                                           p : Quiver.Path a c
                                           f : Quiver.Hom c a
                                           h : Eq (p.cons f).toList Quiver.Path.nil.toList
                                           ⊢ Eq (p.cons f) Quiver.Path.nil
                                         -/
  | _, @cons _ _ _ c _ p f, nil, h => by cases h
                                         /-
                                           🎉 no goals
                                         -/
  | _, @cons _ _ _ c _ p f, @cons _ _ _ t _ C D, h => by
    /-
      V : Type u
      inst✝¹ : Quiver V
      inst✝ : ∀ (a b : V), Subsingleton (Quiver.Hom a b)
      a x✝ c : V
      p : Quiver.Path a c
      f : Quiver.Hom c x✝
      t : V
      C : Quiver.Path a t
      D : Quiver.Hom t x✝
      h : Eq (p.cons f).toList (C.cons D).toList
      ⊢ Eq (p.cons f) (C.cons D)
    -/
    simp only [toList, List.cons.injEq] at h
    /-
      V : Type u
      inst✝¹ : Quiver V
      inst✝ : ∀ (a b : V), Subsingleton (Quiver.Hom a b)
      a x✝ c : V
      p : Quiver.Path a c
      f : Quiver.Hom c x✝
      t : V
      C : Quiver.Path a t
      D : Quiver.Hom t x✝
      h : And (Eq c t) (Eq p.toList C.toList)
      ⊢ Eq (p.cons f) (C.cons D)
    -/
    obtain ⟨rfl, hAC⟩ := h
    /-
      case intro
      V : Type u
      inst✝¹ : Quiver V
      inst✝ : ∀ (a b : V), Subsingleton (Quiver.Hom a b)
      a x✝ c : V
      p : Quiver.Path a c
      f : Quiver.Hom c x✝
      C : Quiver.Path a c
      D : Quiver.Hom c x✝
      hAC : Eq p.toList C.toList
      ⊢ Eq (p.cons f) (C.cons D)
    -/
    simp [toList_injective _ _ hAC, eq_iff_true_of_subsingleton]
    /-
      🎉 no goals
    -/


@[simp]
theorem toList_inj {p q : Path a b} : p.toList = q.toList ↔ p = q :=
  (toList_injective _ _).eq_iff


/-- The image of a path under a prefunctor. -/
def mapPath {a : V} : ∀ {b : V}, Path a b → Path (F.obj a) (F.obj b)
  | _, Path.nil => Path.nil
  | _, Path.cons p e => Path.cons (mapPath p) (F.map e)


@[simp]
theorem mapPath_nil (a : V) : F.mapPath (Path.nil : Path a a) = Path.nil :=
  rfl


@[simp]
theorem mapPath_cons {a b c : V} (p : Path a b) (e : b ⟶ c) :
    F.mapPath (Path.cons p e) = Path.cons (F.mapPath p) (F.map e) :=
  rfl


@[simp]
theorem mapPath_comp {a b : V} (p : Path a b) :
    ∀ {c : V} (q : Path b c), F.mapPath (p.comp q) = (F.mapPath p).comp (F.mapPath q)
  | _, Path.nil => rfl
                           /-
                             V : Type u₁
                             inst✝¹ : Quiver V
                             W : Type u₂
                             inst✝ : Quiver W
                             F : Prefunctor V W
                             a b : V
                             p : Quiver.Path a b
                             c b✝ : V
                             q : Quiver.Path b b✝
                             e : Quiver.Hom b✝ c
                             ⊢ Eq (F.mapPath (p.comp (q.cons e))) ((F.mapPath p).comp (F.mapPath (q.cons e)))
                           -/
  | c, Path.cons q e => by dsimp; rw [mapPath_comp p q]
                                  /-
                                    🎉 no goals
                                  -/


@[simp]
theorem mapPath_toPath {a b : V} (f : a ⟶ b) : F.mapPath f.toPath = (F.map f).toPath :=
  rfl


