/-- An embedding of a complex shape `c : ComplexShape ι` into a complex shape
`c' : ComplexShape ι'` consists of a injective map `f : ι → ι'` which satisfies
a compatibility with respect to the relations `c.Rel` and `c'.Rel`. -/
structure Embedding where
  /-- the map between the underlying types of indices -/
  f : ι → ι'
  injective_f : Function.Injective f
  rel {i₁ i₂ : ι} (h : c.Rel i₁ i₂) : c'.Rel (f i₁) (f i₂)


/-- The opposite embedding in `Embedding c.symm c'.symm` of `e : Embedding c c'`. -/
@[simps]
def op : Embedding c.symm c'.symm where
  f := e.f
  injective_f := e.injective_f
  rel h := e.rel h


/-- An embedding of complex shapes `e` satisfies `e.IsRelIff` if the implication
`e.rel` is an equivalence. -/
class IsRelIff : Prop where
  rel' (i₁ i₂ : ι) (h : c'.Rel (e.f i₁) (e.f i₂)) : c.Rel i₁ i₂


lemma rel_iff [e.IsRelIff] (i₁ i₂ : ι) : c'.Rel (e.f i₁) (e.f i₂) ↔ c.Rel i₁ i₂ := by
  /-
    ι : Type u_1
    ι' : Type u_2
    c : ComplexShape ι
    c' : ComplexShape ι'
    e : c.Embedding c'
    inst✝ : e.IsRelIff
    i₁ i₂ : ι
    ⊢ Iff (c'.Rel (e.f i₁) (e.f i₂)) (c.Rel i₁ i₂)
  -/
  constructor
    /-
      case mp
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      e : c.Embedding c'
      inst✝ : e.IsRelIff
      i₁ i₂ : ι
      ⊢ c'.Rel (e.f i₁) (e.f i₂) → c.Rel i₁ i₂
    -/
  · apply IsRelIff.rel'
    /-
      🎉 no goals
    -/
    /-
      case mpr
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      e : c.Embedding c'
      inst✝ : e.IsRelIff
      i₁ i₂ : ι
      ⊢ c.Rel i₁ i₂ → c'.Rel (e.f i₁) (e.f i₂)
    -/
  · exact e.rel
    /-
      🎉 no goals
    -/


/-- Constructor for embeddings between complex shapes when we have an equivalence
`∀ (i₁ i₂ : ι), c.Rel i₁ i₂ ↔ c'.Rel (f i₁) (f i₂)`. -/
@[simps]
def mk' : Embedding c c' where
  f := f
  injective_f := hf
  rel h := (iff _ _).1 h


instance : (mk' c c' f hf iff).IsRelIff where
  rel' _ _ h := (iff _ _).2 h


/-- The condition that the image of the map `e.f` of an embedding of
complex shapes `e : Embedding c c'` is stable by `c'.next`. -/
class IsTruncGE extends e.IsRelIff : Prop where
  mem_next {j : ι} {k' : ι'} (h : c'.Rel (e.f j) k') :
    ∃ k, e.f k = k'


lemma mem_next [e.IsTruncGE] {j : ι} {k' : ι'} (h : c'.Rel (e.f j) k') : ∃ k, e.f k = k' :=
  IsTruncGE.mem_next h


/-- The condition that the image of the map `e.f` of an embedding of
complex shapes `e : Embedding c c'` is stable by `c'.prev`. -/
class IsTruncLE extends e.IsRelIff : Prop where
  mem_prev {i' : ι'} {j : ι} (h : c'.Rel i' (e.f j)) :
    ∃ i, e.f i = i'


lemma mem_prev [e.IsTruncLE] {i' : ι'} {j : ι} (h : c'.Rel i' (e.f j)) : ∃ i, e.f i = i' :=
  IsTruncLE.mem_prev h


open Classical in
/-- The map `ι' → Option ι` which sends `e.f i` to `some i` and the other elements to `none`. -/
noncomputable def r (i' : ι') : Option ι :=
  if h : ∃ (i : ι), e.f i = i'
  then some h.choose
  else none


lemma r_eq_some {i : ι} {i' : ι'} (hi : e.f i = i') :
    e.r i' = some i := by
  /-
    ι : Type u_1
    ι' : Type u_2
    c : ComplexShape ι
    c' : ComplexShape ι'
    e : c.Embedding c'
    i : ι
    i' : ι'
    hi : Eq (e.f i) i'
    ⊢ Eq (e.r i') (Option.some i)
  -/
  have h : ∃ (i : ι), e.f i = i' := ⟨i, hi⟩
  /-
    ι : Type u_1
    ι' : Type u_2
    c : ComplexShape ι
    c' : ComplexShape ι'
    e : c.Embedding c'
    i : ι
    i' : ι'
    hi : Eq (e.f i) i'
    h : Exists fun i => Eq (e.f i) i'
    ⊢ Eq (e.r i') (Option.some i)
  -/
  have : h.choose = i := e.injective_f (h.choose_spec.trans (hi.symm))
  /-
    ι : Type u_1
    ι' : Type u_2
    c : ComplexShape ι
    c' : ComplexShape ι'
    e : c.Embedding c'
    i : ι
    i' : ι'
    hi : Eq (e.f i) i'
    h : Exists fun i => Eq (e.f i) i'
    this : Eq h.choose i
    ⊢ Eq (e.r i') (Option.some i)
  -/
  dsimp [r]
  /-
    ι : Type u_1
    ι' : Type u_2
    c : ComplexShape ι
    c' : ComplexShape ι'
    e : c.Embedding c'
    i : ι
    i' : ι'
    hi : Eq (e.f i) i'
    h : Exists fun i => Eq (e.f i) i'
    this : Eq h.choose i
    ⊢ Eq (dite (Exists fun i => Eq (e.f i) i') (fun h => Option.some h.choose) fun …
  -/
  rw [dif_pos ⟨i, hi⟩, this]
  /-
    🎉 no goals
  -/


lemma r_eq_none (i' : ι') (hi : ∀ i, e.f i ≠ i') :
    e.r i' = none :=
  dif_neg (by
    /-
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      e : c.Embedding c'
      i' : ι'
      hi : ∀ (i : ι), Ne (e.f i) i'
      ⊢ Not (Exists fun i => Eq (e.f i) i')
    -/
    rintro ⟨i, hi'⟩
    /-
      case intro
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      e : c.Embedding c'
      i' : ι'
      hi : ∀ (i : ι), Ne (e.f i) i'
      i : ι
      hi' : Eq (e.f i) i'
      ⊢ False
    -/
    exact hi i hi')
    /-
      🎉 no goals
    -/


@[simp] lemma r_f (i : ι) : e.r (e.f i) = some i := r_eq_some _ rfl


lemma f_eq_of_r_eq_some {i : ι} {i' : ι'} (hi : e.r i' = some i) :
    e.f i = i' := by
  /-
    ι : Type u_1
    ι' : Type u_2
    c : ComplexShape ι
    c' : ComplexShape ι'
    e : c.Embedding c'
    i : ι
    i' : ι'
    hi : Eq (e.r i') (Option.some i)
    ⊢ Eq (e.f i) i'
  -/
  by_cases h : ∃ (k : ι), e.f k = i'
    /-
      case pos
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      e : c.Embedding c'
      i : ι
      i' : ι'
      hi : Eq (e.r i') (Option.some i)
      h : Exists fun k => Eq (e.f k) i'
      ⊢ Eq (e.f i) i'
    -/
  · obtain ⟨k, rfl⟩ := h
    /-
      case pos.intro
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      e : c.Embedding c'
      i k : ι
      hi : Eq (e.r (e.f k)) (Option.some i)
      ⊢ Eq (e.f i) (e.f k)
    -/
    rw [r_f] at hi
    /-
      case pos.intro
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      e : c.Embedding c'
      i k : ι
      hi : Eq (Option.some k) (Option.some i)
      ⊢ Eq (e.f i) (e.f k)
    -/
    congr 1
    /-
      case pos.intro.e_a
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      e : c.Embedding c'
      i k : ι
      hi : Eq (Option.some k) (Option.some i)
      ⊢ Eq i k
    -/
    simpa using hi.symm
    /-
      🎉 no goals
    -/
    /-
      case neg
      ι : Type u_1
      ι' : Type u_2
      c : ComplexShape ι
      c' : ComplexShape ι'
      e : c.Embedding c'
      i : ι
      i' : ι'
      hi : Eq (e.r i') (Option.some i)
      h : Not (Exists fun k => Eq (e.f k) i')
      ⊢ Eq (e.f i) i'
    -/
  · simp [e.r_eq_none i' (by simpa using h)] at hi
    /-
      🎉 no goals
    -/


/-- The obvious embedding from `up ℕ` to `up ℤ`. -/
@[simps!]
def embeddingUpNat : Embedding (up ℕ) (up ℤ) :=
  Embedding.mk' _ _ (fun n => n)
                     /-
                       ι : Type u_1
                       ι' : Type u_2
                       c : ComplexShape ι
                       c' : ComplexShape ι'
                       x✝¹ x✝ : Nat
                       h : Eq ((fun n => ↑n) x✝¹) ((fun n => ↑n) x✝)
                       ⊢ Eq x✝¹ x✝
                     -/
    (fun _ _ h => by simpa using h)
                     /-
                       🎉 no goals
                     -/
        /-
          ι : Type u_1
          ι' : Type u_2
          c : ComplexShape ι
          c' : ComplexShape ι'
          ⊢ ∀ (i₁ i₂ : Nat), Iff ((ComplexShape.up Nat).Rel i₁ i₂) ((ComplexShape.up Int …
        -/
    (by dsimp; omega)
               /-
                 🎉 no goals
               -/


                                         /-
                                           ι : Type u_1
                                           ι' : Type u_2
                                           c : ComplexShape ι
                                           c' : ComplexShape ι'
                                           ⊢ ComplexShape.embeddingUpNat.IsRelIff
                                         -/
instance : embeddingUpNat.IsRelIff := by dsimp [embeddingUpNat]; infer_instance
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


instance : embeddingUpNat.IsTruncGE where
  mem_next {j _} h := ⟨j + 1, h⟩


/-- The embedding from `down ℕ` to `up ℤ` with sends `n` to `-n`. -/
@[simps!]
def embeddingDownNat : Embedding (down ℕ) (up ℤ) :=
  Embedding.mk' _ _ (fun n => -n)
                     /-
                       ι : Type u_1
                       ι' : Type u_2
                       c : ComplexShape ι
                       c' : ComplexShape ι'
                       x✝¹ x✝ : Nat
                       h : Eq ((fun n => Neg.neg ↑n) x✝¹) ((fun n => Neg.neg ↑n) x✝)
                       ⊢ Eq x✝¹ x✝
                     -/
    (fun _ _ h => by simpa using h)
                     /-
                       🎉 no goals
                     -/
        /-
          ι : Type u_1
          ι' : Type u_2
          c : ComplexShape ι
          c' : ComplexShape ι'
          ⊢ ∀ (i₁ i₂ : Nat), Iff ((ComplexShape.down Nat).Rel i₁ i₂) ((ComplexShape.up I …
        -/
    (by dsimp; omega)
               /-
                 🎉 no goals
               -/


                                           /-
                                             ι : Type u_1
                                             ι' : Type u_2
                                             c : ComplexShape ι
                                             c' : ComplexShape ι'
                                             ⊢ ComplexShape.embeddingDownNat.IsRelIff
                                           -/
instance : embeddingDownNat.IsRelIff := by dsimp [embeddingDownNat]; infer_instance
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


instance : embeddingDownNat.IsTruncLE where
                                 /-
                                   ι : Type u_1
                                   ι' : Type u_2
                                   c : ComplexShape ι
                                   c' : ComplexShape ι'
                                   i : Int
                                   j : Nat
                                   h : (ComplexShape.up Int).Rel i (ComplexShape.embeddingDownNat.f j)
                                   ⊢ Eq (ComplexShape.embeddingDownNat.f (HAdd.hAdd j 1)) i
                                 -/
  mem_prev {i j} h := ⟨j + 1, by dsimp at h ⊢; omega⟩
                                               /-
                                                 🎉 no goals
                                               -/


/-- The embedding from `up ℕ` to `up ℤ` which sends `n : ℕ` to `p + n`. -/
@[simps!]
def embeddingUpIntGE : Embedding (up ℕ) (up ℤ) :=
  Embedding.mk' _ _ (fun n => p + n)
                     /-
                       ι : Type u_1
                       ι' : Type u_2
                       c : ComplexShape ι
                       c' : ComplexShape ι'
                       p : Int
                       x✝¹ x✝ : Nat
                       h : Eq ((fun n => HAdd.hAdd p ↑n) x✝¹) ((fun n => HAdd.hAdd p ↑n) x✝)
                       ⊢ Eq x✝¹ x✝
                     -/
    (fun _ _ h => by dsimp at h; omega)
                                 /-
                                   🎉 no goals
                                 -/
        /-
          ι : Type u_1
          ι' : Type u_2
          c : ComplexShape ι
          c' : ComplexShape ι'
          p : Int
          ⊢ ∀ (i₁ i₂ : Nat), Iff ((ComplexShape.up Nat).Rel i₁ i₂) ((ComplexShape.up Int …
        -/
    (by dsimp; omega)
               /-
                 🎉 no goals
               -/


                                               /-
                                                 ι : Type u_1
                                                 ι' : Type u_2
                                                 c : ComplexShape ι
                                                 c' : ComplexShape ι'
                                                 p : Int
                                                 ⊢ (ComplexShape.embeddingUpIntGE p).IsRelIff
                                               -/
instance : (embeddingUpIntGE p).IsRelIff := by dsimp [embeddingUpIntGE]; infer_instance
                                                                         /-
                                                                           🎉 no goals
                                                                         -/


instance : (embeddingUpIntGE p).IsTruncGE where
                                 /-
                                   ι : Type u_1
                                   ι' : Type u_2
                                   c : ComplexShape ι
                                   c' : ComplexShape ι'
                                   p : Int
                                   j : Nat
                                   x✝ : Int
                                   h : (ComplexShape.up Int).Rel ((ComplexShape.embeddingUpIntGE p).f j) x✝
                                   ⊢ Eq ((ComplexShape.embeddingUpIntGE p).f (HAdd.hAdd j 1)) x✝
                                 -/
  mem_next {j _} h := ⟨j + 1, by dsimp at h ⊢; omega⟩
                                               /-
                                                 🎉 no goals
                                               -/


/-- The embedding from `down ℕ` to `up ℤ` which sends `n : ℕ` to `p - n`. -/
@[simps!]
def embeddingUpIntLE : Embedding (down ℕ) (up ℤ) :=
  Embedding.mk' _ _ (fun n => p - n)
                     /-
                       ι : Type u_1
                       ι' : Type u_2
                       c : ComplexShape ι
                       c' : ComplexShape ι'
                       p : Int
                       x✝¹ x✝ : Nat
                       h : Eq ((fun n => HSub.hSub p ↑n) x✝¹) ((fun n => HSub.hSub p ↑n) x✝)
                       ⊢ Eq x✝¹ x✝
                     -/
    (fun _ _ h => by dsimp at h; omega)
                                 /-
                                   🎉 no goals
                                 -/
        /-
          ι : Type u_1
          ι' : Type u_2
          c : ComplexShape ι
          c' : ComplexShape ι'
          p : Int
          ⊢ ∀ (i₁ i₂ : Nat), Iff ((ComplexShape.down Nat).Rel i₁ i₂) ((ComplexShape.up I …
        -/
    (by dsimp; omega)
               /-
                 🎉 no goals
               -/


                                               /-
                                                 ι : Type u_1
                                                 ι' : Type u_2
                                                 c : ComplexShape ι
                                                 c' : ComplexShape ι'
                                                 p : Int
                                                 ⊢ (ComplexShape.embeddingUpIntLE p).IsRelIff
                                               -/
instance : (embeddingUpIntLE p).IsRelIff := by dsimp [embeddingUpIntLE]; infer_instance
                                                                         /-
                                                                           🎉 no goals
                                                                         -/


instance : (embeddingUpIntLE p).IsTruncLE where
                                 /-
                                   ι : Type u_1
                                   ι' : Type u_2
                                   c : ComplexShape ι
                                   c' : ComplexShape ι'
                                   p x✝ : Int
                                   k : Nat
                                   h : (ComplexShape.up Int).Rel x✝ ((ComplexShape.embeddingUpIntLE p).f k)
                                   ⊢ Eq ((ComplexShape.embeddingUpIntLE p).f (HAdd.hAdd k 1)) x✝
                                 -/
  mem_prev {_ k} h := ⟨k + 1, by dsimp at h ⊢; omega⟩
                                               /-
                                                 🎉 no goals
                                               -/


