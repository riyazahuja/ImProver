/-- The composable arrows associated to a short complex. -/
@[simps!]
def ShortComplex.toComposableArrows (S : ShortComplex C) : ComposableArrows C 2 :=
  ComposableArrows.mk₂ S.f S.g


/-- `F : ComposableArrows C n` is a complex if all compositions of
two consecutive arrows are zero. -/
structure IsComplex : Prop where
  /-- the composition of two consecutive arrows is zero -/
  zero (i : ℕ) (hi : i + 2 ≤ n := by omega) :
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.509, u_1} C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      n : Nat
      S : CategoryTheory.ComposableArrows C n
      i : Nat
      hi : autoParam (LE.le (HAdd.hAdd i 2) n) _auto✝
      ⊢ LE.le i (HAdd.hAdd i 1)
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
    S.map' i (i + 1) ≫ S.map' (i + 1) (i + 2) = 0
                       /-
                         🎉 no goals
                       -/


attribute [reassoc] IsComplex.zero


@[reassoc]
lemma IsComplex.zero' (hS : S.IsComplex) (i j k : ℕ) (hij : i + 1 = j := by omega)
    (hjk : j + 1 = k := by omega) (hk : k ≤ n := by omega) :
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.3156, u_1} C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      n : Nat
      S : CategoryTheory.ComposableArrows C n
      hS : S.IsComplex
      i j k : Nat
      hij : autoParam (Eq (HAdd.hAdd i 1) j) _auto✝
      hjk : autoParam (Eq (HAdd.hAdd j 1) k) _auto✝
      hk : autoParam (LE.le k n) _auto✝
      ⊢ LE.le i j
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
    S.map' i j ≫ S.map' j k = 0 := by
                 /-
                   🎉 no goals
                 -/
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    n : Nat
    S : CategoryTheory.ComposableArrows C n
    hS : S.IsComplex
    i j k : Nat
    hij : autoParam (Eq (HAdd.hAdd i 1) j) _auto✝
    hjk : autoParam (Eq (HAdd.hAdd j 1) k) _auto✝
    hk : autoParam (LE.le k n) _auto✝
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (S.map' i j ⋯ ⋯) (S.map' j k ⋯ hk)) 0
  -/
  subst hij hjk
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    n : Nat
    S : CategoryTheory.ComposableArrows C n
    hS : S.IsComplex
    i : Nat
    hk : autoParam (LE.le (HAdd.hAdd (HAdd.hAdd i 1) 1) n) _auto✝
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (S.map' i (HAdd.hAdd i 1) ⋯ ⋯) (S.map …
  -/
  exact hS.zero i hk
  /-
    🎉 no goals
  -/


lemma isComplex_of_iso {S₁ S₂ : ComposableArrows C n} (e : S₁ ≅ S₂) (h₁ : S₁.IsComplex) :
    S₂.IsComplex where
  zero i hi := by
    rw [← cancel_epi (ComposableArrows.app' e.hom i), comp_zero,
      ← NatTrans.naturality_assoc, ← NatTrans.naturality,
      reassoc_of% (h₁.zero i hi), zero_comp]


lemma isComplex_iff_of_iso {S₁ S₂ : ComposableArrows C n} (e : S₁ ≅ S₂) :
    S₁.IsComplex ↔ S₂.IsComplex :=
  ⟨isComplex_of_iso e, isComplex_of_iso e.symm⟩


lemma isComplex₀ (S : ComposableArrows C 0) : S.IsComplex where
  -- See https://github.com/leanprover/lean4/issues/2862
  -- Without `decide := true`, simp gets stuck at `hi : autoParam False _auto✝`
                  /-
                    C : Type u_1
                    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
                    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                    S : CategoryTheory.ComposableArrows C 0
                    i : Nat
                    hi : autoParam (LE.le (HAdd.hAdd i 2) 0) _auto✝
                    ⊢ Eq (CategoryTheory.CategoryStruct.comp (S.map' i (HAdd.hAdd i 1) ⋯ ⋯) (S.map …
                  -/
  zero i hi := by simp +decide at hi
                  /-
                    🎉 no goals
                  -/


lemma isComplex₁ (S : ComposableArrows C 1) : S.IsComplex where
                  /-
                    C : Type u_1
                    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
                    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                    S : CategoryTheory.ComposableArrows C 1
                    i : Nat
                    hi : autoParam (LE.le (HAdd.hAdd i 2) 1) _auto✝
                    ⊢ Eq (CategoryTheory.CategoryStruct.comp (S.map' i (HAdd.hAdd i 1) ⋯ ⋯) (S.map …
                  -/
  zero i hi := by omega
                  /-
                    🎉 no goals
                  -/


/-- The short complex consisting of maps `S.map' i j` and `S.map' j k` when we know
that `S : ComposableArrows C n` satisfies `S.IsComplex`. -/
abbrev sc' (hS : S.IsComplex) (i j k : ℕ) (hij : i + 1 = j := by omega)
    (hjk : j + 1 = k := by omega) (hk : k ≤ n := by omega) :
    ShortComplex C :=
                   /-
                     C : Type u_1
                     inst✝¹ : CategoryTheory.Category.{?u.16703, u_1} C
                     inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                     n : Nat
                     S : CategoryTheory.ComposableArrows C n
                     hS : S.IsComplex
                     i j k : Nat
                     hij : autoParam (Eq (HAdd.hAdd i 1) j) _auto✝
                     hjk : autoParam (Eq (HAdd.hAdd j 1) k) _auto✝
                     hk : autoParam (LE.le k n) _auto✝
                     ⊢ LE.le i j
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
  ShortComplex.mk (S.map' i j) (S.map' j k) (hS.zero' i j k)
                                             /-
                                               🎉 no goals
                                             -/


/-- The short complex consisting of maps `S.map' i (i + 1)` and `S.map' (i + 1) (i + 2)`
when we know that `S : ComposableArrows C n` satisfies `S.IsComplex`. -/
abbrev sc (hS : S.IsComplex) (i : ℕ) (hi : i + 2 ≤ n := by omega) :
    ShortComplex C :=
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{?u.18053, u_1} C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    n : Nat
    S : CategoryTheory.ComposableArrows C n
    hS : S.IsComplex
    i : Nat
    hi : autoParam (LE.le (HAdd.hAdd i 2) n) _auto✝
    ⊢ Eq (HAdd.hAdd i 1) (HAdd.hAdd i 1)
  -/
  /-
    🎉 no goals
  -/
  /-
    🎉 no goals
  -/
  S.sc' hS i (i + 1) (i + 2)
  /-
    🎉 no goals
  -/


/-- `F : ComposableArrows C n` is exact if it is a complex and that all short
complexes consisting of two consecutive arrows are exact. -/
structure Exact extends S.IsComplex : Prop where
                                                /-
                                                  C : Type u_1
                                                  inst✝¹ : CategoryTheory.Category.{?u.18609, u_1} C
                                                  inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                                                  n : Nat
                                                  S : CategoryTheory.ComposableArrows C n
                                                  toIsComplex : S.IsComplex
                                                  zero : ∀ (i : Nat) (hi : autoParam (LE.le (HAdd.hAdd i 2) n) _auto✝), Eq (Cate …
                                                  i : Nat
                                                  hi : autoParam (LE.le (HAdd.hAdd i 2) n) _auto✝
                                                  ⊢ LE.le (HAdd.hAdd i 2) n
                                                -/
  exact (i : ℕ) (hi : i + 2 ≤ n := by omega) : (S.sc toIsComplex i).Exact
                                                /-
                                                  🎉 no goals
                                                -/


lemma Exact.exact' (hS : S.Exact) (i j k : ℕ) (hij : i + 1 = j := by omega)
    (hjk : j + 1 = k := by omega) (hk : k ≤ n := by omega) :
     /-
       C : Type u_1
       inst✝¹ : CategoryTheory.Category.{?u.19009, u_1} C
       inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
       n : Nat
       S : CategoryTheory.ComposableArrows C n
       hS : S.Exact
       i j k : Nat
       hij : autoParam (Eq (HAdd.hAdd i 1) j) _auto✝
       hjk : autoParam (Eq (HAdd.hAdd j 1) k) _auto✝
       hk : autoParam (LE.le k n) _auto✝
       ⊢ Eq (HAdd.hAdd i 1) j
     -/
     /-
       🎉 no goals
     -/
     /-
       🎉 no goals
     -/
    (S.sc' hS.toIsComplex i j k).Exact := by
     /-
       🎉 no goals
     -/
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    n : Nat
    S : CategoryTheory.ComposableArrows C n
    hS : S.Exact
    i j k : Nat
    hij : autoParam (Eq (HAdd.hAdd i 1) j) _auto✝
    hjk : autoParam (Eq (HAdd.hAdd j 1) k) _auto✝
    hk : autoParam (LE.le k n) _auto✝
    ⊢ (S.sc' ⋯ i j k ⋯ ⋯ ⋯).Exact
  -/
  subst hij hjk
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    n : Nat
    S : CategoryTheory.ComposableArrows C n
    hS : S.Exact
    i : Nat
    hk : autoParam (LE.le (HAdd.hAdd (HAdd.hAdd i 1) 1) n) _auto✝
    ⊢ (S.sc' ⋯ i (HAdd.hAdd i 1) (HAdd.hAdd (HAdd.hAdd i 1) 1) ⋯ ⋯ ⋯).Exact
  -/
  exact hS.exact i hk
  /-
    🎉 no goals
  -/


/-- Functoriality maps for `ComposableArrows.sc'`. -/
@[simps]
def sc'Map {S₁ S₂ : ComposableArrows C n} (φ : S₁ ⟶ S₂) (h₁ : S₁.IsComplex) (h₂ : S₂.IsComplex)
    (i j k : ℕ) (hij : i + 1 = j := by omega)
    (hjk : j + 1 = k := by omega) (hk : k ≤ n := by omega) :
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.19644, u_1} C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      n : Nat
      S S₁ S₂ : CategoryTheory.ComposableArrows C n
      φ : Quiver.Hom S₁ S₂
      h₁ : S₁.IsComplex
      h₂ : S₂.IsComplex
      i j k : Nat
      hij : autoParam (Eq (HAdd.hAdd i 1) j) _auto✝
      hjk : autoParam (Eq (HAdd.hAdd j 1) k) _auto✝
      hk : autoParam (LE.le k n) _auto✝
      ⊢ Eq (HAdd.hAdd i 1) j
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
    S₁.sc' h₁ i j k ⟶ S₂.sc' h₂ i j k where
                      /-
                        🎉 no goals
                      -/
  τ₁ := φ.app _
  τ₂ := φ.app _
  τ₃ := φ.app _


/-- Functoriality maps for `ComposableArrows.sc`. -/
@[simps!]
def scMap {S₁ S₂ : ComposableArrows C n} (φ : S₁ ⟶ S₂) (h₁ : S₁.IsComplex) (h₂ : S₂.IsComplex)
    (i : ℕ) (hi : i + 2 ≤ n := by omega) :
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.33343, u_1} C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      n : Nat
      S S₁ S₂ : CategoryTheory.ComposableArrows C n
      φ : Quiver.Hom S₁ S₂
      h₁ : S₁.IsComplex
      h₂ : S₂.IsComplex
      i : Nat
      hi : autoParam (LE.le (HAdd.hAdd i 2) n) _auto✝
      ⊢ LE.le (HAdd.hAdd i 2) n
    -/
    /-
      🎉 no goals
    -/
    S₁.sc h₁ i ⟶ S₂.sc h₂ i :=
                 /-
                   🎉 no goals
                 -/
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{?u.33343, u_1} C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    n : Nat
    S S₁ S₂ : CategoryTheory.ComposableArrows C n
    φ : Quiver.Hom S₁ S₂
    h₁ : S₁.IsComplex
    h₂ : S₂.IsComplex
    i : Nat
    hi : autoParam (LE.le (HAdd.hAdd i 2) n) _auto✝
    ⊢ Eq (HAdd.hAdd i 1) (HAdd.hAdd i 1)
  -/
  /-
    🎉 no goals
  -/
  /-
    🎉 no goals
  -/
  sc'Map φ h₁ h₂ i (i + 1) (i + 2)
  /-
    🎉 no goals
  -/


/-- The isomorphism `S₁.sc' _ i j k ≅ S₂.sc' _ i j k` induced by an isomorphism `S₁ ≅ S₂`
in `ComposableArrows C n`. -/
@[simps]
def sc'MapIso {S₁ S₂ : ComposableArrows C n} (e : S₁ ≅ S₂)
    (h₁ : S₁.IsComplex) (h₂ : S₂.IsComplex) (i j k : ℕ) (hij : i + 1 = j := by omega)
    (hjk : j + 1 = k := by omega) (hk : k ≤ n := by omega) :
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.34423, u_1} C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      n : Nat
      S S₁ S₂ : CategoryTheory.ComposableArrows C n
      e : CategoryTheory.Iso S₁ S₂
      h₁ : S₁.IsComplex
      h₂ : S₂.IsComplex
      i j k : Nat
      hij : autoParam (Eq (HAdd.hAdd i 1) j) _auto✝
      hjk : autoParam (Eq (HAdd.hAdd j 1) k) _auto✝
      hk : autoParam (LE.le k n) _auto✝
      ⊢ Eq (HAdd.hAdd i 1) j
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
    S₁.sc' h₁ i j k ≅ S₂.sc' h₂ i j k where
                      /-
                        🎉 no goals
                      -/
         /-
           C : Type u_1
           inst✝¹ : CategoryTheory.Category.{?u.34423, u_1} C
           inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
           n : Nat
           S S₁ S₂ : CategoryTheory.ComposableArrows C n
           e : CategoryTheory.Iso S₁ S₂
           h₁ : S₁.IsComplex
           h₂ : S₂.IsComplex
           i j k : Nat
           hij : autoParam (Eq (HAdd.hAdd i 1) j) _auto✝
           hjk : autoParam (Eq (HAdd.hAdd j 1) k) _auto✝
           hk : autoParam (LE.le k n) _auto✝
           ⊢ Eq (HAdd.hAdd i 1) j
         -/
         /-
           🎉 no goals
         -/
         /-
           🎉 no goals
         -/
  hom := sc'Map e.hom h₁ h₂ i j k
         /-
           🎉 no goals
         -/
         /-
           C : Type u_1
           inst✝¹ : CategoryTheory.Category.{?u.34423, u_1} C
           inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
           n : Nat
           S S₁ S₂ : CategoryTheory.ComposableArrows C n
           e : CategoryTheory.Iso S₁ S₂
           h₁ : S₁.IsComplex
           h₂ : S₂.IsComplex
           i j k : Nat
           hij : autoParam (Eq (HAdd.hAdd i 1) j) _auto✝
           hjk : autoParam (Eq (HAdd.hAdd j 1) k) _auto✝
           hk : autoParam (LE.le k n) _auto✝
           ⊢ Eq (HAdd.hAdd i 1) j
         -/
         /-
           🎉 no goals
         -/
         /-
           🎉 no goals
         -/
  inv := sc'Map e.inv h₂ h₁ i j k
         /-
           🎉 no goals
         -/
                   /-
                     C : Type u_1
                     inst✝¹ : CategoryTheory.Category.{?u.34423, u_1} C
                     inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                     n : Nat
                     S S₁ S₂ : CategoryTheory.ComposableArrows C n
                     e : CategoryTheory.Iso S₁ S₂
                     h₁ : S₁.IsComplex
                     h₂ : S₂.IsComplex
                     i j k : Nat
                     hij : autoParam (Eq (HAdd.hAdd i 1) j) _auto✝
                     hjk : autoParam (Eq (HAdd.hAdd j 1) k) _auto✝
                     hk : autoParam (LE.le k n) _auto✝
                     ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.ComposableArrows.sc'M …
                   -/
                                     /-
                                       🎉 no goals
                                     -/
                                     /-
                                       🎉 no goals
                                     -/
  hom_inv_id := by ext <;> dsimp <;> simp
                                     /-
                                       🎉 no goals
                                     -/
                   /-
                     C : Type u_1
                     inst✝¹ : CategoryTheory.Category.{?u.34423, u_1} C
                     inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                     n : Nat
                     S S₁ S₂ : CategoryTheory.ComposableArrows C n
                     e : CategoryTheory.Iso S₁ S₂
                     h₁ : S₁.IsComplex
                     h₂ : S₂.IsComplex
                     i j k : Nat
                     hij : autoParam (Eq (HAdd.hAdd i 1) j) _auto✝
                     hjk : autoParam (Eq (HAdd.hAdd j 1) k) _auto✝
                     hk : autoParam (LE.le k n) _auto✝
                     ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.ComposableArrows.sc'M …
                   -/
                                     /-
                                       🎉 no goals
                                     -/
                                     /-
                                       🎉 no goals
                                     -/
  inv_hom_id := by ext <;> dsimp <;> simp
                                     /-
                                       🎉 no goals
                                     -/


/-- The isomorphism `S₁.sc _ i ≅ S₂.sc _ i` induced by an isomorphism `S₁ ≅ S₂`
in `ComposableArrows C n`. -/
@[simps]
def scMapIso {S₁ S₂ : ComposableArrows C n} (e : S₁ ≅ S₂)
    (h₁ : S₁.IsComplex) (h₂ : S₂.IsComplex)
    (i : ℕ) (hi : i + 2 ≤ n := by omega) :
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.48107, u_1} C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      n : Nat
      S S₁ S₂ : CategoryTheory.ComposableArrows C n
      e : CategoryTheory.Iso S₁ S₂
      h₁ : S₁.IsComplex
      h₂ : S₂.IsComplex
      i : Nat
      hi : autoParam (LE.le (HAdd.hAdd i 2) n) _auto✝
      ⊢ LE.le (HAdd.hAdd i 2) n
    -/
    /-
      🎉 no goals
    -/
    S₁.sc h₁ i ≅ S₂.sc h₂ i where
                 /-
                   🎉 no goals
                 -/
         /-
           C : Type u_1
           inst✝¹ : CategoryTheory.Category.{?u.48107, u_1} C
           inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
           n : Nat
           S S₁ S₂ : CategoryTheory.ComposableArrows C n
           e : CategoryTheory.Iso S₁ S₂
           h₁ : S₁.IsComplex
           h₂ : S₂.IsComplex
           i : Nat
           hi : autoParam (LE.le (HAdd.hAdd i 2) n) _auto✝
           ⊢ LE.le (HAdd.hAdd i 2) n
         -/
  hom := scMap e.hom h₁ h₂ i
         /-
           🎉 no goals
         -/
         /-
           C : Type u_1
           inst✝¹ : CategoryTheory.Category.{?u.48107, u_1} C
           inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
           n : Nat
           S S₁ S₂ : CategoryTheory.ComposableArrows C n
           e : CategoryTheory.Iso S₁ S₂
           h₁ : S₁.IsComplex
           h₂ : S₂.IsComplex
           i : Nat
           hi : autoParam (LE.le (HAdd.hAdd i 2) n) _auto✝
           ⊢ LE.le (HAdd.hAdd i 2) n
         -/
  inv := scMap e.inv h₂ h₁ i
         /-
           🎉 no goals
         -/
                   /-
                     C : Type u_1
                     inst✝¹ : CategoryTheory.Category.{?u.48107, u_1} C
                     inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                     n : Nat
                     S S₁ S₂ : CategoryTheory.ComposableArrows C n
                     e : CategoryTheory.Iso S₁ S₂
                     h₁ : S₁.IsComplex
                     h₂ : S₂.IsComplex
                     i : Nat
                     hi : autoParam (LE.le (HAdd.hAdd i 2) n) _auto✝
                     ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.ComposableArrows.scMa …
                   -/
                                     /-
                                       🎉 no goals
                                     -/
                                     /-
                                       🎉 no goals
                                     -/
  hom_inv_id := by ext <;> dsimp <;> simp
                                     /-
                                       🎉 no goals
                                     -/
                   /-
                     C : Type u_1
                     inst✝¹ : CategoryTheory.Category.{?u.48107, u_1} C
                     inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                     n : Nat
                     S S₁ S₂ : CategoryTheory.ComposableArrows C n
                     e : CategoryTheory.Iso S₁ S₂
                     h₁ : S₁.IsComplex
                     h₂ : S₂.IsComplex
                     i : Nat
                     hi : autoParam (LE.le (HAdd.hAdd i 2) n) _auto✝
                     ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.ComposableArrows.scMa …
                   -/
                                     /-
                                       🎉 no goals
                                     -/
                                     /-
                                       🎉 no goals
                                     -/
  inv_hom_id := by ext <;> dsimp <;> simp
                                     /-
                                       🎉 no goals
                                     -/


lemma exact_of_iso {S₁ S₂ : ComposableArrows C n} (e : S₁ ≅ S₂) (h₁ : S₁.Exact) :
    S₂.Exact where
  toIsComplex := isComplex_of_iso e h₁.toIsComplex
                                           /-
                                             C : Type u_1
                                             inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
                                             inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                                             n : Nat
                                             S₁ S₂ : CategoryTheory.ComposableArrows C n
                                             e : CategoryTheory.Iso S₁ S₂
                                             h₁ : S₁.Exact
                                             i : Nat
                                             hi : autoParam (LE.le (HAdd.hAdd i 2) n) _auto✝
                                             ⊢ LE.le (HAdd.hAdd i 2) n
                                           -/
  exact i hi := ShortComplex.exact_of_iso (scMapIso e h₁.toIsComplex
                                           /-
                                             🎉 no goals
                                           -/
    (isComplex_of_iso e h₁.toIsComplex) i) (h₁.exact i hi)


lemma exact_iff_of_iso {S₁ S₂ : ComposableArrows C n} (e : S₁ ≅ S₂) :
    S₁.Exact ↔ S₂.Exact :=
  ⟨exact_of_iso e, exact_of_iso e.symm⟩


lemma exact₀ (S : ComposableArrows C 0) : S.Exact where
  toIsComplex := S.isComplex₀
  -- See https://github.com/leanprover/lean4/issues/2862
                   /-
                     C : Type u_1
                     inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
                     inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                     S : CategoryTheory.ComposableArrows C 0
                     i : Nat
                     hi : autoParam (LE.le (HAdd.hAdd i 2) 0) _auto✝
                     ⊢ (S.sc ⋯ i ⋯).Exact
                   -/
  exact i hi := by simp at hi
                   /-
                     🎉 no goals
                   -/


lemma exact₁ (S : ComposableArrows C 1) : S.Exact where
  toIsComplex := S.isComplex₁
                   /-
                     C : Type u_1
                     inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
                     inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                     S : CategoryTheory.ComposableArrows C 1
                     i : Nat
                     hi : autoParam (LE.le (HAdd.hAdd i 2) 1) _auto✝
                     ⊢ (S.sc ⋯ i ⋯).Exact
                   -/
  exact i hi := by exfalso; omega
                            /-
                              🎉 no goals
                            -/


lemma isComplex₂_iff (S : ComposableArrows C 2) :
                  /-
                    C : Type u_1
                    inst✝¹ : CategoryTheory.Category.{?u.58843, u_1} C
                    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                    n : Nat
                    S✝ : CategoryTheory.ComposableArrows C n
                    S : CategoryTheory.ComposableArrows C 2
                    ⊢ LE.le 0 1
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
    S.IsComplex ↔ S.map' 0 1 ≫ S.map' 1 2 = 0 := by
                               /-
                                 🎉 no goals
                               -/
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    S : CategoryTheory.ComposableArrows C 2
    ⊢ Iff S.IsComplex (Eq (CategoryTheory.CategoryStruct.comp (S.map' 0 1 ⋯ ⋯) (S. …
  -/
  constructor
    /-
      case mp
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      S : CategoryTheory.ComposableArrows C 2
      ⊢ S.IsComplex → Eq (CategoryTheory.CategoryStruct.comp (S.map' 0 1 ⋯ ⋯) (S.map …
    -/
  · intro h
    /-
      case mp
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      S : CategoryTheory.ComposableArrows C 2
      h : S.IsComplex
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (S.map' 0 1 ⋯ ⋯) (S.map' 1 2 ⋯ ⋯)) 0
    -/
    exact h.zero 0 (by omega)
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      S : CategoryTheory.ComposableArrows C 2
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (S.map' 0 1 ⋯ ⋯) (S.map' 1 2 ⋯ ⋯)) 0  …
    -/
  · intro h
    /-
      case mpr
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      S : CategoryTheory.ComposableArrows C 2
      h : Eq (CategoryTheory.CategoryStruct.comp (S.map' 0 1 ⋯ ⋯) (S.map' 1 2 ⋯ ⋯)) 0
      ⊢ S.IsComplex
    -/
    refine IsComplex.mk (fun i hi => ?_)
    /-
      case mpr
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      S : CategoryTheory.ComposableArrows C 2
      h : Eq (CategoryTheory.CategoryStruct.comp (S.map' 0 1 ⋯ ⋯) (S.map' 1 2 ⋯ ⋯)) 0
      i : Nat
      hi : autoParam (LE.le (HAdd.hAdd i 2) 2) _auto✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (S.map' i (HAdd.hAdd i 1) ⋯ ⋯) (S.map …
    -/
    obtain rfl : i = 0 := by omega
    /-
      case mpr
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      S : CategoryTheory.ComposableArrows C 2
      h : Eq (CategoryTheory.CategoryStruct.comp (S.map' 0 1 ⋯ ⋯) (S.map' 1 2 ⋯ ⋯)) 0
      hi : autoParam (LE.le (HAdd.hAdd 0 2) 2) _auto✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (S.map' 0 (HAdd.hAdd 0 1) ⋯ ⋯) (S.map …
    -/
    exact h
    /-
      🎉 no goals
    -/


                                                    /-
                                                      C : Type u_1
                                                      inst✝¹ : CategoryTheory.Category.{?u.63054, u_1} C
                                                      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                                                      n : Nat
                                                      S✝ : CategoryTheory.ComposableArrows C n
                                                      S : CategoryTheory.ComposableArrows C 2
                                                      ⊢ LE.le 0 1
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
lemma isComplex₂_mk (S : ComposableArrows C 2) (w : S.map' 0 1 ≫ S.map' 1 2 = 0) :
                                                                 /-
                                                                   🎉 no goals
                                                                 -/
    S.IsComplex :=
  S.isComplex₂_iff.2 w


set_option simprocs false in
lemma _root_.CategoryTheory.ShortComplex.isComplex_toComposableArrows (S : ShortComplex C) :
    S.toComposableArrows.IsComplex :=
                      /-
                        C : Type u_1
                        inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
                        inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                        S : CategoryTheory.ShortComplex C
                        ⊢ Eq (CategoryTheory.CategoryStruct.comp (S.toComposableArrows.map' 0 1 ⋯ ⋯) ( …
                      -/
  isComplex₂_mk _ (by simp)
                      /-
                        🎉 no goals
                      -/


lemma exact₂_iff (S : ComposableArrows C 2) (hS : S.IsComplex) :
               /-
                 C : Type u_1
                 inst✝¹ : CategoryTheory.Category.{?u.67926, u_1} C
                 inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                 n : Nat
                 S✝ : CategoryTheory.ComposableArrows C n
                 S : CategoryTheory.ComposableArrows C 2
                 hS : S.IsComplex
                 ⊢ Eq (HAdd.hAdd 0 1) 1
               -/
               /-
                 🎉 no goals
               -/
               /-
                 🎉 no goals
               -/
    S.Exact ↔ (S.sc' hS 0 1 2).Exact := by
               /-
                 🎉 no goals
               -/
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    S : CategoryTheory.ComposableArrows C 2
    hS : S.IsComplex
    ⊢ Iff S.Exact (S.sc' hS 0 1 2 ⋯ ⋯ ⋯).Exact
  -/
  constructor
    /-
      case mp
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      S : CategoryTheory.ComposableArrows C 2
      hS : S.IsComplex
      ⊢ S.Exact → (S.sc' hS 0 1 2 ⋯ ⋯ ⋯).Exact
    -/
  · intro h
    /-
      case mp
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      S : CategoryTheory.ComposableArrows C 2
      hS : S.IsComplex
      h : S.Exact
      ⊢ (S.sc' hS 0 1 2 ⋯ ⋯ ⋯).Exact
    -/
    exact h.exact 0 (by omega)
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      S : CategoryTheory.ComposableArrows C 2
      hS : S.IsComplex
      ⊢ (S.sc' hS 0 1 2 ⋯ ⋯ ⋯).Exact → S.Exact
    -/
  · intro h
    /-
      case mpr
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      S : CategoryTheory.ComposableArrows C 2
      hS : S.IsComplex
      h : (S.sc' hS 0 1 2 ⋯ ⋯ ⋯).Exact
      ⊢ S.Exact
    -/
    refine Exact.mk hS (fun i hi => ?_)
    /-
      case mpr
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      S : CategoryTheory.ComposableArrows C 2
      hS : S.IsComplex
      h : (S.sc' hS 0 1 2 ⋯ ⋯ ⋯).Exact
      i : Nat
      hi : autoParam (LE.le (HAdd.hAdd i 2) 2) _auto✝
      ⊢ (S.sc hS i ⋯).Exact
    -/
    obtain rfl : i = 0 := by omega
    /-
      case mpr
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      S : CategoryTheory.ComposableArrows C 2
      hS : S.IsComplex
      h : (S.sc' hS 0 1 2 ⋯ ⋯ ⋯).Exact
      hi : autoParam (LE.le (HAdd.hAdd 0 2) 2) _auto✝
      ⊢ (S.sc hS 0 ⋯).Exact
    -/
    exact h
    /-
      🎉 no goals
    -/


                                                /-
                                                  C : Type u_1
                                                  inst✝¹ : CategoryTheory.Category.{?u.69217, u_1} C
                                                  inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                                                  n : Nat
                                                  S✝ : CategoryTheory.ComposableArrows C n
                                                  S : CategoryTheory.ComposableArrows C 2
                                                  ⊢ LE.le 0 1
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
lemma exact₂_mk (S : ComposableArrows C 2) (w : S.map' 0 1 ≫ S.map' 1 2 = 0)
                                                             /-
                                                               🎉 no goals
                                                             -/
    (h : (ShortComplex.mk _ _ w).Exact) : S.Exact :=
  (S.exact₂_iff (S.isComplex₂_mk w)).2 h


lemma _root_.CategoryTheory.ShortComplex.Exact.exact_toComposableArrows
    {S : ShortComplex C} (hS : S.Exact) :
    S.toComposableArrows.Exact :=
  exact₂_mk _ _ hS


lemma _root_.CategoryTheory.ShortComplex.exact_iff_exact_toComposableArrows
    (S : ShortComplex C) :
    S.Exact ↔ S.toComposableArrows.Exact :=
  (S.toComposableArrows.exact₂_iff S.isComplex_toComposableArrows).symm


lemma exact_iff_δ₀ (S : ComposableArrows C (n + 2)) :
                    /-
                      C : Type u_1
                      inst✝¹ : CategoryTheory.Category.{?u.72115, u_1} C
                      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                      n : Nat
                      S✝ : CategoryTheory.ComposableArrows C n
                      S : CategoryTheory.ComposableArrows C (HAdd.hAdd n 2)
                      ⊢ LE.le 0 1
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
    S.Exact ↔ (mk₂ (S.map' 0 1) (S.map' 1 2)).Exact ∧ S.δ₀.Exact := by
                                 /-
                                   🎉 no goals
                                 -/
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    n : Nat
    S : CategoryTheory.ComposableArrows C (HAdd.hAdd n 2)
    ⊢ Iff S.Exact (And (CategoryTheory.ComposableArrows.mk₂ (S.map' 0 1 ⋯ ⋯) (S.ma …
  -/
  constructor
    /-
      case mp
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      n : Nat
      S : CategoryTheory.ComposableArrows C (HAdd.hAdd n 2)
      ⊢ S.Exact → And (CategoryTheory.ComposableArrows.mk₂ (S.map' 0 1 ⋯ ⋯) (S.map'  …
    -/
  · intro h
    /-
      case mp
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      n : Nat
      S : CategoryTheory.ComposableArrows C (HAdd.hAdd n 2)
      h : S.Exact
      ⊢ And (CategoryTheory.ComposableArrows.mk₂ (S.map' 0 1 ⋯ ⋯) (S.map' 1 2 ⋯ ⋯)). …
    -/
    constructor
      /-
        case mp.left
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
        inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
        n : Nat
        S : CategoryTheory.ComposableArrows C (HAdd.hAdd n 2)
        h : S.Exact
        ⊢ (CategoryTheory.ComposableArrows.mk₂ (S.map' 0 1 ⋯ ⋯) (S.map' 1 2 ⋯ ⋯)).Exact
      -/
    · rw [exact₂_iff]; swap
        /-
          case mp.left.hS
          C : Type u_1
          inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
          n : Nat
          S : CategoryTheory.ComposableArrows C (HAdd.hAdd n 2)
          h : S.Exact
          ⊢ (CategoryTheory.ComposableArrows.mk₂ (S.map' 0 1 ⋯ ⋯) (S.map' 1 2 ⋯ ⋯)).IsCo …
        -/
      · rw [isComplex₂_iff]
        /-
          case mp.left.hS
          C : Type u_1
          inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
          n : Nat
          S : CategoryTheory.ComposableArrows C (HAdd.hAdd n 2)
          h : S.Exact
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.ComposableArrows.mk₂ …
        -/
        exact h.toIsComplex.zero 0
        /-
          🎉 no goals
        -/
      /-
        case mp.left
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
        inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
        n : Nat
        S : CategoryTheory.ComposableArrows C (HAdd.hAdd n 2)
        h : S.Exact
        ⊢ ((CategoryTheory.ComposableArrows.mk₂ (S.map' 0 1 ⋯ ⋯) (S.map' 1 2 ⋯ ⋯)).sc' …
      -/
      exact h.exact 0 (by omega)
      /-
        🎉 no goals
      -/
    · exact Exact.mk (IsComplex.mk (fun i hi => h.toIsComplex.zero (i + 1)))
        (fun i hi => h.exact (i + 1))
    /-
      case mpr
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      n : Nat
      S : CategoryTheory.ComposableArrows C (HAdd.hAdd n 2)
      ⊢ And (CategoryTheory.ComposableArrows.mk₂ (S.map' 0 1 ⋯ ⋯) (S.map' 1 2 ⋯ ⋯)). …
    -/
  · rintro ⟨h, h₀⟩
    /-
      case mpr.intro
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      n : Nat
      S : CategoryTheory.ComposableArrows C (HAdd.hAdd n 2)
      h : (CategoryTheory.ComposableArrows.mk₂ (S.map' 0 1 ⋯ ⋯) (S.map' 1 2 ⋯ ⋯)).Ex …
      h₀ : S.δ₀.Exact
      ⊢ S.Exact
    -/
    refine Exact.mk (IsComplex.mk (fun i hi => ?_)) (fun i hi => ?_)
      /-
        case mpr.intro.refine_1
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
        inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
        n : Nat
        S : CategoryTheory.ComposableArrows C (HAdd.hAdd n 2)
        h : (CategoryTheory.ComposableArrows.mk₂ (S.map' 0 1 ⋯ ⋯) (S.map' 1 2 ⋯ ⋯)).Ex …
        h₀ : S.δ₀.Exact
        i : Nat
        hi : autoParam (LE.le (HAdd.hAdd i 2) (HAdd.hAdd n 2)) _auto✝
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (S.map' i (HAdd.hAdd i 1) ⋯ ⋯) (S.map …
      -/
    · obtain _ | i := i
        /-
          case mpr.intro.refine_1.zero
          C : Type u_1
          inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
          n : Nat
          S : CategoryTheory.ComposableArrows C (HAdd.hAdd n 2)
          h : (CategoryTheory.ComposableArrows.mk₂ (S.map' 0 1 ⋯ ⋯) (S.map' 1 2 ⋯ ⋯)).Ex …
          h₀ : S.δ₀.Exact
          hi : autoParam (LE.le (HAdd.hAdd 0 2) (HAdd.hAdd n 2)) _auto✝
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (S.map' 0 (HAdd.hAdd 0 1) ⋯ ⋯) (S.map …
        -/
      · exact h.toIsComplex.zero 0
        /-
          🎉 no goals
        -/
        /-
          case mpr.intro.refine_1.succ
          C : Type u_1
          inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
          n : Nat
          S : CategoryTheory.ComposableArrows C (HAdd.hAdd n 2)
          h : (CategoryTheory.ComposableArrows.mk₂ (S.map' 0 1 ⋯ ⋯) (S.map' 1 2 ⋯ ⋯)).Ex …
          h₀ : S.δ₀.Exact
          i : Nat
          hi : autoParam (LE.le (HAdd.hAdd (HAdd.hAdd i 1) 2) (HAdd.hAdd n 2)) _auto✝
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (S.map' (HAdd.hAdd i 1) (HAdd.hAdd (H …
        -/
      · exact h₀.toIsComplex.zero i
        /-
          🎉 no goals
        -/
      /-
        case mpr.intro.refine_2
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
        inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
        n : Nat
        S : CategoryTheory.ComposableArrows C (HAdd.hAdd n 2)
        h : (CategoryTheory.ComposableArrows.mk₂ (S.map' 0 1 ⋯ ⋯) (S.map' 1 2 ⋯ ⋯)).Ex …
        h₀ : S.δ₀.Exact
        i : Nat
        hi : autoParam (LE.le (HAdd.hAdd i 2) (HAdd.hAdd n 2)) _auto✝
        ⊢ (S.sc ⋯ i ⋯).Exact
      -/
    · obtain _ | i := i
        /-
          case mpr.intro.refine_2.zero
          C : Type u_1
          inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
          n : Nat
          S : CategoryTheory.ComposableArrows C (HAdd.hAdd n 2)
          h : (CategoryTheory.ComposableArrows.mk₂ (S.map' 0 1 ⋯ ⋯) (S.map' 1 2 ⋯ ⋯)).Ex …
          h₀ : S.δ₀.Exact
          hi : autoParam (LE.le (HAdd.hAdd 0 2) (HAdd.hAdd n 2)) _auto✝
          ⊢ (S.sc ⋯ 0 ⋯).Exact
        -/
      · exact h.exact 0
        /-
          🎉 no goals
        -/
        /-
          case mpr.intro.refine_2.succ
          C : Type u_1
          inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
          n : Nat
          S : CategoryTheory.ComposableArrows C (HAdd.hAdd n 2)
          h : (CategoryTheory.ComposableArrows.mk₂ (S.map' 0 1 ⋯ ⋯) (S.map' 1 2 ⋯ ⋯)).Ex …
          h₀ : S.δ₀.Exact
          i : Nat
          hi : autoParam (LE.le (HAdd.hAdd (HAdd.hAdd i 1) 2) (HAdd.hAdd n 2)) _auto✝
          ⊢ (S.sc ⋯ (HAdd.hAdd i 1) ⋯).Exact
        -/
      · exact h₀.exact i
        /-
          🎉 no goals
        -/


lemma Exact.δ₀ {S : ComposableArrows C (n + 2)} (hS : S.Exact) :
    S.δ₀.Exact := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    n : Nat
    S : CategoryTheory.ComposableArrows C (HAdd.hAdd n 2)
    hS : S.Exact
    ⊢ S.δ₀.Exact
  -/
  rw [exact_iff_δ₀] at hS
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    n : Nat
    S : CategoryTheory.ComposableArrows C (HAdd.hAdd n 2)
    hS : And (CategoryTheory.ComposableArrows.mk₂ (S.map' 0 1 ⋯ ⋯) (S.map' 1 2 ⋯ ⋯ …
    ⊢ S.δ₀.Exact
  -/
  exact hS.2
  /-
    🎉 no goals
  -/


/-- If `S : ComposableArrows C (n + 2)` is such that the first two arrows form
an exact sequence and that the tail `S.δ₀` is exact, then `S` is also exact.
See `ShortComplex.SnakeInput.snake_lemma` in `Algebra.Homology.ShortComplex.SnakeLemma`
for a use of this lemma. -/
lemma exact_of_δ₀ {S : ComposableArrows C (n + 2)}
               /-
                 C : Type u_1
                 inst✝¹ : CategoryTheory.Category.{?u.82562, u_1} C
                 inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                 n : Nat
                 S✝ : CategoryTheory.ComposableArrows C n
                 S : CategoryTheory.ComposableArrows C (HAdd.hAdd n 2)
                 ⊢ LE.le 0 1
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
    (h : (mk₂ (S.map' 0 1) (S.map' 1 2)).Exact) (h₀ : S.δ₀.Exact) : S.Exact := by
                            /-
                              🎉 no goals
                            -/
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    n : Nat
    S : CategoryTheory.ComposableArrows C (HAdd.hAdd n 2)
    h : (CategoryTheory.ComposableArrows.mk₂ (S.map' 0 1 ⋯ ⋯) (S.map' 1 2 ⋯ ⋯)).Ex …
    h₀ : S.δ₀.Exact
    ⊢ S.Exact
  -/
  rw [exact_iff_δ₀]
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    n : Nat
    S : CategoryTheory.ComposableArrows C (HAdd.hAdd n 2)
    h : (CategoryTheory.ComposableArrows.mk₂ (S.map' 0 1 ⋯ ⋯) (S.map' 1 2 ⋯ ⋯)).Ex …
    h₀ : S.δ₀.Exact
    ⊢ And (CategoryTheory.ComposableArrows.mk₂ (S.map' 0 1 ⋯ ⋯) (S.map' 1 2 ⋯ ⋯)). …
  -/
                  /-
                    🎉 no goals
                  -/
  constructor <;> assumption
                  /-
                    🎉 no goals
                  -/


lemma exact_iff_δlast {n : ℕ} (S : ComposableArrows C (n + 2)) :
                                    /-
                                      C : Type u_1
                                      inst✝¹ : CategoryTheory.Category.{?u.83958, u_1} C
                                      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                                      n✝ : Nat
                                      S✝ : CategoryTheory.ComposableArrows C n✝
                                      n : Nat
                                      S : CategoryTheory.ComposableArrows C (HAdd.hAdd n 2)
                                      ⊢ LE.le n (HAdd.hAdd n 1)
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
    S.Exact ↔ S.δlast.Exact ∧ (mk₂ (S.map' n (n + 1)) (S.map' (n + 1) (n + 2))).Exact := by
                                                       /-
                                                         🎉 no goals
                                                       -/
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    n : Nat
    S : CategoryTheory.ComposableArrows C (HAdd.hAdd n 2)
    ⊢ Iff S.Exact (And S.δlast.Exact (CategoryTheory.ComposableArrows.mk₂ (S.map'  …
  -/
  constructor
    /-
      case mp
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      n : Nat
      S : CategoryTheory.ComposableArrows C (HAdd.hAdd n 2)
      ⊢ S.Exact → And S.δlast.Exact (CategoryTheory.ComposableArrows.mk₂ (S.map' n ( …
    -/
  · intro h
    /-
      case mp
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      n : Nat
      S : CategoryTheory.ComposableArrows C (HAdd.hAdd n 2)
      h : S.Exact
      ⊢ And S.δlast.Exact (CategoryTheory.ComposableArrows.mk₂ (S.map' n (HAdd.hAdd  …
    -/
    constructor
    · exact Exact.mk (IsComplex.mk (fun i hi => h.toIsComplex.zero i))
        (fun i hi => h.exact i)
      /-
        case mp.right
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
        inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
        n : Nat
        S : CategoryTheory.ComposableArrows C (HAdd.hAdd n 2)
        h : S.Exact
        ⊢ (CategoryTheory.ComposableArrows.mk₂ (S.map' n (HAdd.hAdd n 1) ⋯ ⋯) (S.map'  …
      -/
    · rw [exact₂_iff]; swap
        /-
          case mp.right.hS
          C : Type u_1
          inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
          n : Nat
          S : CategoryTheory.ComposableArrows C (HAdd.hAdd n 2)
          h : S.Exact
          ⊢ (CategoryTheory.ComposableArrows.mk₂ (S.map' n (HAdd.hAdd n 1) ⋯ ⋯) (S.map'  …
        -/
      · rw [isComplex₂_iff]
        /-
          case mp.right.hS
          C : Type u_1
          inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
          n : Nat
          S : CategoryTheory.ComposableArrows C (HAdd.hAdd n 2)
          h : S.Exact
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.ComposableArrows.mk₂ …
        -/
        exact h.toIsComplex.zero n
        /-
          🎉 no goals
        -/
      /-
        case mp.right
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
        inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
        n : Nat
        S : CategoryTheory.ComposableArrows C (HAdd.hAdd n 2)
        h : S.Exact
        ⊢ ((CategoryTheory.ComposableArrows.mk₂ (S.map' n (HAdd.hAdd n 1) ⋯ ⋯) (S.map' …
      -/
      exact h.exact n (by omega)
      /-
        🎉 no goals
      -/
    /-
      case mpr
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      n : Nat
      S : CategoryTheory.ComposableArrows C (HAdd.hAdd n 2)
      ⊢ And S.δlast.Exact (CategoryTheory.ComposableArrows.mk₂ (S.map' n (HAdd.hAdd  …
    -/
  · rintro ⟨h, h'⟩
    /-
      case mpr.intro
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      n : Nat
      S : CategoryTheory.ComposableArrows C (HAdd.hAdd n 2)
      h : S.δlast.Exact
      h' : (CategoryTheory.ComposableArrows.mk₂ (S.map' n (HAdd.hAdd n 1) ⋯ ⋯) (S.ma …
      ⊢ S.Exact
    -/
    refine Exact.mk (IsComplex.mk (fun i hi => ?_)) (fun i hi => ?_)
      /-
        case mpr.intro.refine_1
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
        inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
        n : Nat
        S : CategoryTheory.ComposableArrows C (HAdd.hAdd n 2)
        h : S.δlast.Exact
        h' : (CategoryTheory.ComposableArrows.mk₂ (S.map' n (HAdd.hAdd n 1) ⋯ ⋯) (S.ma …
        i : Nat
        hi : autoParam (LE.le (HAdd.hAdd i 2) (HAdd.hAdd n 2)) _auto✝
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (S.map' i (HAdd.hAdd i 1) ⋯ ⋯) (S.map …
      -/
    · simp only [Nat.add_le_add_iff_right] at hi
      /-
        case mpr.intro.refine_1
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
        inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
        n : Nat
        S : CategoryTheory.ComposableArrows C (HAdd.hAdd n 2)
        h : S.δlast.Exact
        h' : (CategoryTheory.ComposableArrows.mk₂ (S.map' n (HAdd.hAdd n 1) ⋯ ⋯) (S.ma …
        i : Nat
        hi✝ : autoParam (LE.le (HAdd.hAdd i 2) (HAdd.hAdd n 2)) _auto✝
        hi : autoParam (LE.le i n) _auto✝
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (S.map' i (HAdd.hAdd i 1) ⋯ ⋯) (S.map …
      -/
      obtain hi | rfl := hi.lt_or_eq
        /-
          case mpr.intro.refine_1.inl
          C : Type u_1
          inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
          n : Nat
          S : CategoryTheory.ComposableArrows C (HAdd.hAdd n 2)
          h : S.δlast.Exact
          h' : (CategoryTheory.ComposableArrows.mk₂ (S.map' n (HAdd.hAdd n 1) ⋯ ⋯) (S.ma …
          i : Nat
          hi✝¹ : autoParam (LE.le (HAdd.hAdd i 2) (HAdd.hAdd n 2)) _auto✝
          hi✝ : autoParam (LE.le i n) _auto✝
          hi : LT.lt i n
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (S.map' i (HAdd.hAdd i 1) ⋯ ⋯) (S.map …
        -/
      · exact h.toIsComplex.zero i
        /-
          🎉 no goals
        -/
        /-
          case mpr.intro.refine_1.inr
          C : Type u_1
          inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
          i : Nat
          S : CategoryTheory.ComposableArrows C (HAdd.hAdd i 2)
          h : S.δlast.Exact
          h' : (CategoryTheory.ComposableArrows.mk₂ (S.map' i (HAdd.hAdd i 1) ⋯ ⋯) (S.ma …
          hi✝ : autoParam (LE.le (HAdd.hAdd i 2) (HAdd.hAdd i 2)) _auto✝
          hi : autoParam (LE.le i i) _auto✝
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (S.map' i (HAdd.hAdd i 1) ⋯ ⋯) (S.map …
        -/
      · exact h'.toIsComplex.zero 0
        /-
          🎉 no goals
        -/
      /-
        case mpr.intro.refine_2
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
        inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
        n : Nat
        S : CategoryTheory.ComposableArrows C (HAdd.hAdd n 2)
        h : S.δlast.Exact
        h' : (CategoryTheory.ComposableArrows.mk₂ (S.map' n (HAdd.hAdd n 1) ⋯ ⋯) (S.ma …
        i : Nat
        hi : autoParam (LE.le (HAdd.hAdd i 2) (HAdd.hAdd n 2)) _auto✝
        ⊢ (S.sc ⋯ i ⋯).Exact
      -/
    · simp only [Nat.add_le_add_iff_right] at hi
      /-
        case mpr.intro.refine_2
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
        inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
        n : Nat
        S : CategoryTheory.ComposableArrows C (HAdd.hAdd n 2)
        h : S.δlast.Exact
        h' : (CategoryTheory.ComposableArrows.mk₂ (S.map' n (HAdd.hAdd n 1) ⋯ ⋯) (S.ma …
        i : Nat
        hi✝ : autoParam (LE.le (HAdd.hAdd i 2) (HAdd.hAdd n 2)) _auto✝
        hi : autoParam (LE.le i n) _auto✝
        ⊢ (S.sc ⋯ i ⋯).Exact
      -/
      obtain hi | rfl := hi.lt_or_eq
        /-
          case mpr.intro.refine_2.inl
          C : Type u_1
          inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
          n : Nat
          S : CategoryTheory.ComposableArrows C (HAdd.hAdd n 2)
          h : S.δlast.Exact
          h' : (CategoryTheory.ComposableArrows.mk₂ (S.map' n (HAdd.hAdd n 1) ⋯ ⋯) (S.ma …
          i : Nat
          hi✝¹ : autoParam (LE.le (HAdd.hAdd i 2) (HAdd.hAdd n 2)) _auto✝
          hi✝ : autoParam (LE.le i n) _auto✝
          hi : LT.lt i n
          ⊢ (S.sc ⋯ i ⋯).Exact
        -/
      · exact h.exact i
        /-
          🎉 no goals
        -/
        /-
          case mpr.intro.refine_2.inr
          C : Type u_1
          inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
          i : Nat
          S : CategoryTheory.ComposableArrows C (HAdd.hAdd i 2)
          h : S.δlast.Exact
          h' : (CategoryTheory.ComposableArrows.mk₂ (S.map' i (HAdd.hAdd i 1) ⋯ ⋯) (S.ma …
          hi✝ : autoParam (LE.le (HAdd.hAdd i 2) (HAdd.hAdd i 2)) _auto✝
          hi : autoParam (LE.le i i) _auto✝
          ⊢ (S.sc ⋯ i ⋯).Exact
        -/
      · exact h'.exact 0
        /-
          🎉 no goals
        -/


lemma Exact.δlast {S : ComposableArrows C (n + 2)} (hS : S.Exact) :
    S.δlast.Exact := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    n : Nat
    S : CategoryTheory.ComposableArrows C (HAdd.hAdd n 2)
    hS : S.Exact
    ⊢ S.δlast.Exact
  -/
  rw [exact_iff_δlast] at hS
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    n : Nat
    S : CategoryTheory.ComposableArrows C (HAdd.hAdd n 2)
    hS : And S.δlast.Exact (CategoryTheory.ComposableArrows.mk₂ (S.map' n (HAdd.hA …
    ⊢ S.δlast.Exact
  -/
  exact hS.1
  /-
    🎉 no goals
  -/


lemma exact_of_δlast {n : ℕ} (S : ComposableArrows C (n + 2))
                                     /-
                                       C : Type u_1
                                       inst✝¹ : CategoryTheory.Category.{?u.96929, u_1} C
                                       inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                                       n✝ : Nat
                                       S✝ : CategoryTheory.ComposableArrows C n✝
                                       n : Nat
                                       S : CategoryTheory.ComposableArrows C (HAdd.hAdd n 2)
                                       h₁ : S.δlast.Exact
                                       ⊢ LE.le n (HAdd.hAdd n 1)
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
    (h₁ : S.δlast.Exact) (h₂ : (mk₂ (S.map' n (n + 1)) (S.map' (n + 1) (n + 2))).Exact) :
                                                        /-
                                                          🎉 no goals
                                                        -/
    S.Exact := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    n : Nat
    S : CategoryTheory.ComposableArrows C (HAdd.hAdd n 2)
    h₁ : S.δlast.Exact
    h₂ : (CategoryTheory.ComposableArrows.mk₂ (S.map' n (HAdd.hAdd n 1) ⋯ ⋯) (S.ma …
    ⊢ S.Exact
  -/
  rw [exact_iff_δlast]
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    n : Nat
    S : CategoryTheory.ComposableArrows C (HAdd.hAdd n 2)
    h₁ : S.δlast.Exact
    h₂ : (CategoryTheory.ComposableArrows.mk₂ (S.map' n (HAdd.hAdd n 1) ⋯ ⋯) (S.ma …
    ⊢ And S.δlast.Exact (CategoryTheory.ComposableArrows.mk₂ (S.map' n (HAdd.hAdd  …
  -/
                  /-
                    🎉 no goals
                  -/
  constructor <;> assumption
                  /-
                    🎉 no goals
                  -/


