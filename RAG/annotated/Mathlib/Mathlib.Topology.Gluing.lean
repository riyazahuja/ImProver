/-- A family of gluing data consists of
1. An index type `J`
2. An object `U i` for each `i : J`.
3. An object `V i j` for each `i j : J`.
  (Note that this is `J × J → TopCat` rather than `J → J → TopCat` to connect to the
  limits library easier.)
4. An open embedding `f i j : V i j ⟶ U i` for each `i j : ι`.
5. A transition map `t i j : V i j ⟶ V j i` for each `i j : ι`.
such that
6. `f i i` is an isomorphism.
7. `t i i` is the identity.
8. `V i j ×[U i] V i k ⟶ V i j ⟶ V j i` factors through `V j k ×[U j] V j i ⟶ V j i` via some
    `t' : V i j ×[U i] V i k ⟶ V j k ×[U j] V j i`.
    (This merely means that `V i j ∩ V i k ⊆ t i j ⁻¹' (V j i ∩ V j k)`.)
9. `t' i j k ≫ t' j k i ≫ t' k i j = 𝟙 _`.

We can then glue the topological spaces `U i` together by identifying `V i j` with `V j i`, such
that the `U i`'s are open subspaces of the glued space.

Most of the times it would be easier to use the constructor `TopCat.GlueData.mk'` where the
conditions are stated in a less categorical way.
-/
-- Porting note (https://github.com/leanprover-community/mathlib4/issues/5171): removed @[nolint has_nonempty_instance]
structure GlueData extends CategoryTheory.GlueData TopCat where
  f_open : ∀ i j, IsOpenEmbedding (f i j)
  f_mono i j := (TopCat.mono_iff_injective _).mpr (f_open i j).isEmbedding.injective


local notation "𝖣" => D.toGlueData


theorem π_surjective : Function.Surjective 𝖣.π :=
  (TopCat.epi_iff_surjective 𝖣.π).mp inferInstance


theorem isOpen_iff (U : Set 𝖣.glued) : IsOpen U ↔ ∀ i, IsOpen (𝖣.ι i ⁻¹' U) := by
  /-
    D : TopCat.GlueData
    U : Set ↑D.glued
    ⊢ Iff (IsOpen U) (∀ (i : D.J), IsOpen (Set.preimage (⇑(D.ι i)) U))
  -/
  delta CategoryTheory.GlueData.ι
  /-
    D : TopCat.GlueData
    U : Set ↑D.glued
    ⊢ Iff (IsOpen U) (∀ (i : D.J), IsOpen (Set.preimage (⇑(CategoryTheory.Limits.M …
  -/
  simp_rw [← Multicoequalizer.ι_sigmaπ 𝖣.diagram]
  /-
    D : TopCat.GlueData
    U : Set ↑D.glued
    ⊢ Iff (IsOpen U) (∀ (i : D.J), IsOpen (Set.preimage (⇑(CategoryTheory.Category …
  -/
  rw [← (homeoOfIso (Multicoequalizer.isoCoequalizer 𝖣.diagram).symm).isOpen_preimage]
  /-
    D : TopCat.GlueData
    U : Set ↑D.glued
    ⊢ Iff (IsOpen (Set.preimage (⇑(TopCat.homeoOfIso (CategoryTheory.Limits.Multic …
  -/
  rw [coequalizer_isOpen_iff, colimit_isOpen_iff.{u}]
  dsimp only [GlueData.diagram_l, GlueData.diagram_left, GlueData.diagram_r, GlueData.diagram_right,
    parallelPair_obj_one]
  /-
    D : TopCat.GlueData
    U : Set ↑D.glued
    ⊢ Iff (∀ (j : CategoryTheory.Discrete D.J), IsOpen (Set.preimage (⇑(CategoryTh …
  -/
  constructor
    /-
      case mp
      D : TopCat.GlueData
      U : Set ↑D.glued
      ⊢ (∀ (j : CategoryTheory.Discrete D.J), IsOpen (Set.preimage (⇑(CategoryTheory …
    -/
  · intro h j; exact h ⟨j⟩
               /-
                 🎉 no goals
               -/
    /-
      case mpr
      D : TopCat.GlueData
      U : Set ↑D.glued
      ⊢ (∀ (i : D.J), IsOpen (Set.preimage (⇑(CategoryTheory.CategoryStruct.comp (Ca …
    -/
  · intro h j; cases j; apply h
                        /-
                          🎉 no goals
                        -/


theorem ι_jointly_surjective (x : 𝖣.glued) : ∃ (i : _) (y : D.U i), 𝖣.ι i y = x :=
  𝖣.ι_jointly_surjective (forget TopCat) x


/-- An equivalence relation on `Σ i, D.U i` that holds iff `𝖣 .ι i x = 𝖣 .ι j y`.
See `TopCat.GlueData.ι_eq_iff_rel`.
-/
def Rel (a b : Σ i, ((D.U i : TopCat) : Type _)) : Prop :=
  a = b ∨ ∃ x : D.V (a.1, b.1), D.f _ _ x = a.2 ∧ D.f _ _ (D.t _ _ x) = b.2


theorem rel_equiv : Equivalence D.Rel :=
  ⟨fun x => Or.inl (refl x), by
    /-
      D : TopCat.GlueData
      ⊢ ∀ {x y : Sigma fun i => ↑(D.U i)}, D.Rel x y → D.Rel y x
    -/
    rintro a b (⟨⟨⟩⟩ | ⟨x, e₁, e₂⟩)
    /-
      case inl.refl
      D : TopCat.GlueData
      a : Sigma fun i => ↑(D.U i)
      ⊢ D.Rel a a
    -/
    exacts [Or.inl rfl, Or.inr ⟨D.t _ _ x, e₂, by rw [← e₁, D.t_inv_apply]⟩], by
    /-
      🎉 no goals
    -/
    /-
      D : TopCat.GlueData
      ⊢ ∀ {x y z : Sigma fun i => ↑(D.U i)}, D.Rel x y → D.Rel y z → D.Rel x z
    -/
    rintro ⟨i, a⟩ ⟨j, b⟩ ⟨k, c⟩ (⟨⟨⟩⟩ | ⟨x, e₁, e₂⟩)
      /-
        case mk.mk.mk.inl.refl
        D : TopCat.GlueData
        i : D.J
        a : ↑(D.U i)
        k : D.J
        c : ↑(D.U k)
        ⊢ D.Rel ⟨i, a⟩ ⟨k, c⟩ → D.Rel ⟨i, a⟩ ⟨k, c⟩
      -/
    · exact id
      /-
        🎉 no goals
      -/
    /-
      case mk.mk.mk.inr.intro.intro
      D : TopCat.GlueData
      i : D.J
      a : ↑(D.U i)
      j : D.J
      b : ↑(D.U j)
      k : D.J
      c : ↑(D.U k)
      x : ↑(D.V { fst := ⟨i, a⟩.fst, snd := ⟨j, b⟩.fst })
      e₁ : Eq ((D.f ⟨i, a⟩.fst ⟨j, b⟩.fst) x) ⟨i, a⟩.snd
      e₂ : Eq ((D.f ⟨j, b⟩.fst ⟨i, a⟩.fst) ((D.t ⟨i, a⟩.fst ⟨j, b⟩.fst) x)) ⟨j, b⟩.snd
      ⊢ D.Rel ⟨j, b⟩ ⟨k, c⟩ → D.Rel ⟨i, a⟩ ⟨k, c⟩
    -/
    rintro (⟨⟨⟩⟩ | ⟨y, e₃, e₄⟩)
      /-
        case mk.mk.mk.inr.intro.intro.inl.refl
        D : TopCat.GlueData
        i : D.J
        a : ↑(D.U i)
        j : D.J
        b : ↑(D.U j)
        x : ↑(D.V { fst := ⟨i, a⟩.fst, snd := ⟨j, b⟩.fst })
        e₁ : Eq ((D.f ⟨i, a⟩.fst ⟨j, b⟩.fst) x) ⟨i, a⟩.snd
        e₂ : Eq ((D.f ⟨j, b⟩.fst ⟨i, a⟩.fst) ((D.t ⟨i, a⟩.fst ⟨j, b⟩.fst) x)) ⟨j, b⟩.snd
        ⊢ D.Rel ⟨i, a⟩ ⟨j, b⟩
      -/
    · exact Or.inr ⟨x, e₁, e₂⟩
      /-
        🎉 no goals
      -/
    /-
      case mk.mk.mk.inr.intro.intro.inr.intro.intro
      D : TopCat.GlueData
      i : D.J
      a : ↑(D.U i)
      j : D.J
      b : ↑(D.U j)
      k : D.J
      c : ↑(D.U k)
      x : ↑(D.V { fst := ⟨i, a⟩.fst, snd := ⟨j, b⟩.fst })
      e₁ : Eq ((D.f ⟨i, a⟩.fst ⟨j, b⟩.fst) x) ⟨i, a⟩.snd
      e₂ : Eq ((D.f ⟨j, b⟩.fst ⟨i, a⟩.fst) ((D.t ⟨i, a⟩.fst ⟨j, b⟩.fst) x)) ⟨j, b⟩.snd
      y : ↑(D.V { fst := ⟨j, b⟩.fst, snd := ⟨k, c⟩.fst })
      e₃ : Eq ((D.f ⟨j, b⟩.fst ⟨k, c⟩.fst) y) ⟨j, b⟩.snd
      e₄ : Eq ((D.f ⟨k, c⟩.fst ⟨j, b⟩.fst) ((D.t ⟨j, b⟩.fst ⟨k, c⟩.fst) y)) ⟨k, c⟩.snd
      ⊢ D.Rel ⟨i, a⟩ ⟨k, c⟩
    -/
    let z := (pullbackIsoProdSubtype (D.f j i) (D.f j k)).inv ⟨⟨_, _⟩, e₂.trans e₃.symm⟩
    have eq₁ : (D.t j i) ((pullback.fst _ _ : _ /-(D.f j k)-/ ⟶ D.V (j, i)) z) = x := by
      dsimp only [coe_of, z]
      erw [pullbackIsoProdSubtype_inv_fst_apply, D.t_inv_apply]-- now `erw` after https://github.com/leanprover-community/mathlib4/pull/13170
    /-
      case mk.mk.mk.inr.intro.intro.inr.intro.intro
      D : TopCat.GlueData
      i : D.J
      a : ↑(D.U i)
      j : D.J
      b : ↑(D.U j)
      k : D.J
      c : ↑(D.U k)
      x : ↑(D.V { fst := ⟨i, a⟩.fst, snd := ⟨j, b⟩.fst })
      e₁ : Eq ((D.f ⟨i, a⟩.fst ⟨j, b⟩.fst) x) ⟨i, a⟩.snd
      e₂ : Eq ((D.f ⟨j, b⟩.fst ⟨i, a⟩.fst) ((D.t ⟨i, a⟩.fst ⟨j, b⟩.fst) x)) ⟨j, b⟩.snd
      y : ↑(D.V { fst := ⟨j, b⟩.fst, snd := ⟨k, c⟩.fst })
      e₃ : Eq ((D.f ⟨j, b⟩.fst ⟨k, c⟩.fst) y) ⟨j, b⟩.snd
      e₄ : Eq ((D.f ⟨k, c⟩.fst ⟨j, b⟩.fst) ((D.t ⟨j, b⟩.fst ⟨k, c⟩.fst) y)) ⟨k, c⟩.snd
      z : ↑(CategoryTheory.Limits.pullback (D.f j i) (D.f j k)) := (TopCat.pullbackI …
      eq₁ : Eq ((D.t j i) ((CategoryTheory.Limits.pullback.fst (D.f j i) (D.f j k))  …
      ⊢ D.Rel ⟨i, a⟩ ⟨k, c⟩
    -/
    have eq₂ : (pullback.snd _ _ : _ ⟶ D.V _) z = y := pullbackIsoProdSubtype_inv_snd_apply _ _ _
    /-
      case mk.mk.mk.inr.intro.intro.inr.intro.intro
      D : TopCat.GlueData
      i : D.J
      a : ↑(D.U i)
      j : D.J
      b : ↑(D.U j)
      k : D.J
      c : ↑(D.U k)
      x : ↑(D.V { fst := ⟨i, a⟩.fst, snd := ⟨j, b⟩.fst })
      e₁ : Eq ((D.f ⟨i, a⟩.fst ⟨j, b⟩.fst) x) ⟨i, a⟩.snd
      e₂ : Eq ((D.f ⟨j, b⟩.fst ⟨i, a⟩.fst) ((D.t ⟨i, a⟩.fst ⟨j, b⟩.fst) x)) ⟨j, b⟩.snd
      y : ↑(D.V { fst := ⟨j, b⟩.fst, snd := ⟨k, c⟩.fst })
      e₃ : Eq ((D.f ⟨j, b⟩.fst ⟨k, c⟩.fst) y) ⟨j, b⟩.snd
      e₄ : Eq ((D.f ⟨k, c⟩.fst ⟨j, b⟩.fst) ((D.t ⟨j, b⟩.fst ⟨k, c⟩.fst) y)) ⟨k, c⟩.snd
      z : ↑(CategoryTheory.Limits.pullback (D.f j i) (D.f j k)) := (TopCat.pullbackI …
      eq₁ : Eq ((D.t j i) ((CategoryTheory.Limits.pullback.fst (D.f j i) (D.f j k))  …
      eq₂ : Eq ((CategoryTheory.Limits.pullback.snd (D.f j i) (D.f j k)) z) y
      ⊢ D.Rel ⟨i, a⟩ ⟨k, c⟩
    -/
    clear_value z
    /-
      case mk.mk.mk.inr.intro.intro.inr.intro.intro
      D : TopCat.GlueData
      i : D.J
      a : ↑(D.U i)
      j : D.J
      b : ↑(D.U j)
      k : D.J
      c : ↑(D.U k)
      x : ↑(D.V { fst := ⟨i, a⟩.fst, snd := ⟨j, b⟩.fst })
      e₁ : Eq ((D.f ⟨i, a⟩.fst ⟨j, b⟩.fst) x) ⟨i, a⟩.snd
      e₂ : Eq ((D.f ⟨j, b⟩.fst ⟨i, a⟩.fst) ((D.t ⟨i, a⟩.fst ⟨j, b⟩.fst) x)) ⟨j, b⟩.snd
      y : ↑(D.V { fst := ⟨j, b⟩.fst, snd := ⟨k, c⟩.fst })
      e₃ : Eq ((D.f ⟨j, b⟩.fst ⟨k, c⟩.fst) y) ⟨j, b⟩.snd
      e₄ : Eq ((D.f ⟨k, c⟩.fst ⟨j, b⟩.fst) ((D.t ⟨j, b⟩.fst ⟨k, c⟩.fst) y)) ⟨k, c⟩.snd
      z : ↑(CategoryTheory.Limits.pullback (D.f j i) (D.f j k))
      eq₁ : Eq ((D.t j i) ((CategoryTheory.Limits.pullback.fst (D.f j i) (D.f j k))  …
      eq₂ : Eq ((CategoryTheory.Limits.pullback.snd (D.f j i) (D.f j k)) z) y
      ⊢ D.Rel ⟨i, a⟩ ⟨k, c⟩
    -/
    right
    /-
      case mk.mk.mk.inr.intro.intro.inr.intro.intro.h
      D : TopCat.GlueData
      i : D.J
      a : ↑(D.U i)
      j : D.J
      b : ↑(D.U j)
      k : D.J
      c : ↑(D.U k)
      x : ↑(D.V { fst := ⟨i, a⟩.fst, snd := ⟨j, b⟩.fst })
      e₁ : Eq ((D.f ⟨i, a⟩.fst ⟨j, b⟩.fst) x) ⟨i, a⟩.snd
      e₂ : Eq ((D.f ⟨j, b⟩.fst ⟨i, a⟩.fst) ((D.t ⟨i, a⟩.fst ⟨j, b⟩.fst) x)) ⟨j, b⟩.snd
      y : ↑(D.V { fst := ⟨j, b⟩.fst, snd := ⟨k, c⟩.fst })
      e₃ : Eq ((D.f ⟨j, b⟩.fst ⟨k, c⟩.fst) y) ⟨j, b⟩.snd
      e₄ : Eq ((D.f ⟨k, c⟩.fst ⟨j, b⟩.fst) ((D.t ⟨j, b⟩.fst ⟨k, c⟩.fst) y)) ⟨k, c⟩.snd
      z : ↑(CategoryTheory.Limits.pullback (D.f j i) (D.f j k))
      eq₁ : Eq ((D.t j i) ((CategoryTheory.Limits.pullback.fst (D.f j i) (D.f j k))  …
      eq₂ : Eq ((CategoryTheory.Limits.pullback.snd (D.f j i) (D.f j k)) z) y
      ⊢ Exists fun x => And (Eq ((D.f ⟨i, a⟩.fst ⟨k, c⟩.fst) x) ⟨i, a⟩.snd) (Eq ((D. …
    -/
    use (pullback.fst _ _ : _ ⟶ D.V (i, k)) (D.t' _ _ _ z)
    /-
      case h
      D : TopCat.GlueData
      i : D.J
      a : ↑(D.U i)
      j : D.J
      b : ↑(D.U j)
      k : D.J
      c : ↑(D.U k)
      x : ↑(D.V { fst := ⟨i, a⟩.fst, snd := ⟨j, b⟩.fst })
      e₁ : Eq ((D.f ⟨i, a⟩.fst ⟨j, b⟩.fst) x) ⟨i, a⟩.snd
      e₂ : Eq ((D.f ⟨j, b⟩.fst ⟨i, a⟩.fst) ((D.t ⟨i, a⟩.fst ⟨j, b⟩.fst) x)) ⟨j, b⟩.snd
      y : ↑(D.V { fst := ⟨j, b⟩.fst, snd := ⟨k, c⟩.fst })
      e₃ : Eq ((D.f ⟨j, b⟩.fst ⟨k, c⟩.fst) y) ⟨j, b⟩.snd
      e₄ : Eq ((D.f ⟨k, c⟩.fst ⟨j, b⟩.fst) ((D.t ⟨j, b⟩.fst ⟨k, c⟩.fst) y)) ⟨k, c⟩.snd
      z : ↑(CategoryTheory.Limits.pullback (D.f j i) (D.f j k))
      eq₁ : Eq ((D.t j i) ((CategoryTheory.Limits.pullback.fst (D.f j i) (D.f j k))  …
      eq₂ : Eq ((CategoryTheory.Limits.pullback.snd (D.f j i) (D.f j k)) z) y
      ⊢ And (Eq ((D.f ⟨i, a⟩.fst ⟨k, c⟩.fst) ((CategoryTheory.Limits.pullback.fst (D …
    -/
    dsimp only at *
    /-
      case h
      D : TopCat.GlueData
      i : D.J
      a : ↑(D.U i)
      j : D.J
      b : ↑(D.U j)
      k : D.J
      c : ↑(D.U k)
      x : ↑(D.V { fst := ⟨i, a⟩.fst, snd := ⟨j, b⟩.fst })
      e₁ : Eq ((D.f i j) x) a
      e₂ : Eq ((D.f j i) ((D.t i j) x)) b
      y : ↑(D.V { fst := ⟨j, b⟩.fst, snd := ⟨k, c⟩.fst })
      e₃ : Eq ((D.f j k) y) b
      e₄ : Eq ((D.f k j) ((D.t j k) y)) c
      z : ↑(CategoryTheory.Limits.pullback (D.f j i) (D.f j k))
      eq₁ : Eq ((D.t j i) ((CategoryTheory.Limits.pullback.fst (D.f j i) (D.f j k))  …
      eq₂ : Eq ((CategoryTheory.Limits.pullback.snd (D.f j i) (D.f j k)) z) y
      ⊢ And (Eq ((D.f i k) ((CategoryTheory.Limits.pullback.fst (D.f i k) (D.f i j)) …
    -/
    substs eq₁ eq₂ e₁ e₃ e₄
    have h₁ : D.t' j i k ≫ pullback.fst _ _ ≫ D.f i k = pullback.fst _ _ ≫ D.t j i ≫ D.f i j := by
      rw [← 𝖣.t_fac_assoc]; congr 1; exact pullback.condition
    have h₂ : D.t' j i k ≫ pullback.fst _ _ ≫ D.t i k ≫ D.f k i =
        pullback.snd _ _ ≫ D.t j k ≫ D.f k j := by
      rw [← 𝖣.t_fac_assoc]
      apply @Epi.left_cancellation _ _ _ _ (D.t' k j i)
      rw [𝖣.cocycle_assoc, 𝖣.t_fac_assoc, 𝖣.t_inv_assoc]
      exact pullback.condition.symm
    /-
      case h
      D : TopCat.GlueData
      i j k : D.J
      z : ↑(CategoryTheory.Limits.pullback (D.f j i) (D.f j k))
      e₂ : Eq ((D.f j i) ((D.t i j) ((D.t j i) ((CategoryTheory.Limits.pullback.fst  …
      h₁ : Eq (CategoryTheory.CategoryStruct.comp (D.t' j i k) (CategoryTheory.Categ …
      h₂ : Eq (CategoryTheory.CategoryStruct.comp (D.t' j i k) (CategoryTheory.Categ …
      ⊢ And (Eq ((D.f i k) ((CategoryTheory.Limits.pullback.fst (D.f i k) (D.f i j)) …
    -/
    exact ⟨ContinuousMap.congr_fun h₁ z, ContinuousMap.congr_fun h₂ z⟩⟩
    /-
      🎉 no goals
    -/


theorem eqvGen_of_π_eq
    -- Porting note: was `{x y : ∐ D.U} (h : 𝖣.π x = 𝖣.π y)`
    {x y : sigmaObj (β := D.toGlueData.J) (C := TopCat) D.toGlueData.U}
    (h : 𝖣.π x = 𝖣.π y) :
    Relation.EqvGen
      -- Porting note: was (Types.CoequalizerRel 𝖣.diagram.fstSigmaMap 𝖣.diagram.sndSigmaMap)
      (Types.CoequalizerRel
        (X := sigmaObj (β := D.toGlueData.diagram.L) (C := TopCat) (D.toGlueData.diagram).left)
        (Y := sigmaObj (β := D.toGlueData.diagram.R) (C := TopCat) (D.toGlueData.diagram).right)
        𝖣.diagram.fstSigmaMap 𝖣.diagram.sndSigmaMap)
      x y := by
  /-
    D : TopCat.GlueData
    x y : ↑(CategoryTheory.Limits.sigmaObj D.U)
    h : Eq (D.π x) (D.π y)
    ⊢ Relation.EqvGen (CategoryTheory.Limits.Types.CoequalizerRel ⇑D.diagram.fstSi …
  -/
  delta GlueData.π Multicoequalizer.sigmaπ at h
  -- Porting note: inlined `inferInstance` instead of leaving as a side goal.
  replace h := (TopCat.mono_iff_injective (Multicoequalizer.isoCoequalizer 𝖣.diagram).inv).mp
    inferInstance h
  /-
    D : TopCat.GlueData
    x y : ↑(CategoryTheory.Limits.sigmaObj D.U)
    h : Eq ((CategoryTheory.Limits.coequalizer.π D.diagram.fstSigmaMap D.diagram.s …
    ⊢ Relation.EqvGen (CategoryTheory.Limits.Types.CoequalizerRel ⇑D.diagram.fstSi …
  -/
  let diagram := parallelPair 𝖣.diagram.fstSigmaMap 𝖣.diagram.sndSigmaMap ⋙ forget _
  have : colimit.ι diagram one x = colimit.ι diagram one y := by
    dsimp only [coequalizer.π, ContinuousMap.toFun_eq_coe] at h
    rw [← ι_preservesColimitIso_hom, forget_map_eq_coe, types_comp_apply, h]
    simp
  have :
    (colimit.ι diagram _ ≫ colim.map _ ≫ (colimit.isoColimitCocone _).hom) _ =
      (colimit.ι diagram _ ≫ colim.map _ ≫ (colimit.isoColimitCocone _).hom) _ :=
    (congr_arg
        (colim.map (diagramIsoParallelPair diagram).hom ≫
          (colimit.isoColimitCocone (Types.coequalizerColimit _ _)).hom)
        this :
      _)
  -- Porting note: was
  -- simp only [eqToHom_refl, types_comp_apply, colimit.ι_map_assoc,
  --   diagramIsoParallelPair_hom_app, colimit.isoColimitCocone_ι_hom, types_id_apply] at this
  -- See https://github.com/leanprover-community/mathlib4/issues/5026
  rw [colimit.ι_map_assoc, diagramIsoParallelPair_hom_app, eqToHom_refl,
    colimit.isoColimitCocone_ι_hom, types_comp_apply, types_id_apply, types_comp_apply,
    types_id_apply] at this
  /-
    D : TopCat.GlueData
    x y : ↑(CategoryTheory.Limits.sigmaObj D.U)
    h : Eq ((CategoryTheory.Limits.coequalizer.π D.diagram.fstSigmaMap D.diagram.s …
    diagram : CategoryTheory.Functor CategoryTheory.Limits.WalkingParallelPair (Ty …
    this✝ : Eq (CategoryTheory.Limits.colimit.ι diagram CategoryTheory.Limits.Walk …
    this : Eq ((CategoryTheory.Limits.Types.coequalizerColimit (diagram.map Catego …
    ⊢ Relation.EqvGen (CategoryTheory.Limits.Types.CoequalizerRel ⇑D.diagram.fstSi …
  -/
  exact Quot.eq.1 this
  /-
    🎉 no goals
  -/


theorem ι_eq_iff_rel (i j : D.J) (x : D.U i) (y : D.U j) :
    𝖣.ι i x = 𝖣.ι j y ↔ D.Rel ⟨i, x⟩ ⟨j, y⟩ := by
  /-
    D : TopCat.GlueData
    i j : D.J
    x : ↑(D.U i)
    y : ↑(D.U j)
    ⊢ Iff (Eq ((D.ι i) x) ((D.ι j) y)) (D.Rel ⟨i, x⟩ ⟨j, y⟩)
  -/
  constructor
    /-
      case mp
      D : TopCat.GlueData
      i j : D.J
      x : ↑(D.U i)
      y : ↑(D.U j)
      ⊢ Eq ((D.ι i) x) ((D.ι j) y) → D.Rel ⟨i, x⟩ ⟨j, y⟩
    -/
  · delta GlueData.ι
    /-
      case mp
      D : TopCat.GlueData
      i j : D.J
      x : ↑(D.U i)
      y : ↑(D.U j)
      ⊢ Eq ((CategoryTheory.Limits.Multicoequalizer.π D.diagram i) x) ((CategoryTheo …
    -/
    simp_rw [← Multicoequalizer.ι_sigmaπ]
    /-
      case mp
      D : TopCat.GlueData
      i j : D.J
      x : ↑(D.U i)
      y : ↑(D.U j)
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Sigma.ι D.dia …
    -/
    intro h
    rw [←
      show _ = Sigma.mk i x from ConcreteCategory.congr_hom (sigmaIsoSigma.{_, u} D.U).inv_hom_id _]
    rw [←
      show _ = Sigma.mk j y from ConcreteCategory.congr_hom (sigmaIsoSigma.{_, u} D.U).inv_hom_id _]
    /-
      case mp
      D : TopCat.GlueData
      i j : D.J
      x : ↑(D.U i)
      y : ↑(D.U j)
      h : Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Sigma.ι D.d …
      ⊢ D.Rel ((CategoryTheory.CategoryStruct.comp (TopCat.sigmaIsoSigma D.U).inv (T …
    -/
    change InvImage D.Rel (sigmaIsoSigma.{_, u} D.U).hom _ _
    /-
      case mp
      D : TopCat.GlueData
      i j : D.J
      x : ↑(D.U i)
      y : ↑(D.U j)
      h : Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Sigma.ι D.d …
      ⊢ InvImage D.Rel (⇑(TopCat.sigmaIsoSigma D.U).hom) ((TopCat.sigmaIsoSigma D.U) …
    -/
    rw [← (InvImage.equivalence _ _ D.rel_equiv).eqvGen_iff]
    /-
      case mp
      D : TopCat.GlueData
      i j : D.J
      x : ↑(D.U i)
      y : ↑(D.U j)
      h : Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Sigma.ι D.d …
      ⊢ Relation.EqvGen (InvImage D.Rel ⇑(TopCat.sigmaIsoSigma D.U).hom) ((TopCat.si …
    -/
    refine Relation.EqvGen.mono ?_ (D.eqvGen_of_π_eq h : _)
    /-
      case mp
      D : TopCat.GlueData
      i j : D.J
      x : ↑(D.U i)
      y : ↑(D.U j)
      h : Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Sigma.ι D.d …
      ⊢ ∀ (a b : ↑(CategoryTheory.Limits.sigmaObj D.U)), CategoryTheory.Limits.Types …
    -/
    rintro _ _ ⟨x⟩
    obtain ⟨⟨⟨i, j⟩, y⟩, rfl⟩ :=
      (ConcreteCategory.bijective_of_isIso (sigmaIsoSigma.{u, u} _).inv).2 x
    /-
      case mp.Rel.intro.mk.mk
      D : TopCat.GlueData
      i✝ j✝ : D.J
      x : ↑(D.U i✝)
      y✝ : ↑(D.U j✝)
      h : Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Sigma.ι D.d …
      i j : D.J
      y : ↑(D.diagram.left { fst := i, snd := j })
      ⊢ InvImage D.Rel (⇑(TopCat.sigmaIsoSigma D.U).hom) (D.diagram.fstSigmaMap ((Ca …
    -/
    unfold InvImage MultispanIndex.fstSigmaMap MultispanIndex.sndSigmaMap
    /-
      case mp.Rel.intro.mk.mk
      D : TopCat.GlueData
      i✝ j✝ : D.J
      x : ↑(D.U i✝)
      y✝ : ↑(D.U j✝)
      h : Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Sigma.ι D.d …
      i j : D.J
      y : ↑(D.diagram.left { fst := i, snd := j })
      ⊢ D.Rel ((TopCat.sigmaIsoSigma D.U).hom ((CategoryTheory.Limits.Sigma.desc fun …
    -/
    simp only [forget_map_eq_coe]
    erw [TopCat.comp_app, sigmaIsoSigma_inv_apply, ← comp_apply, ← comp_apply,
      colimit.ι_desc_assoc, ← comp_apply, ← comp_apply, colimit.ι_desc_assoc]
      -- previous line now `erw` after https://github.com/leanprover-community/mathlib4/pull/13170
    /-
      case mp.Rel.intro.mk.mk
      D : TopCat.GlueData
      i✝ j✝ : D.J
      x : ↑(D.U i✝)
      y✝ : ↑(D.U j✝)
      h : Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Sigma.ι D.d …
      i j : D.J
      y : ↑(D.diagram.left { fst := i, snd := j })
      ⊢ D.Rel ((CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits.Cofan.mk  …
    -/
    erw [sigmaIsoSigma_hom_ι_apply, sigmaIsoSigma_hom_ι_apply]
    /-
      case mp.Rel.intro.mk.mk
      D : TopCat.GlueData
      i✝ j✝ : D.J
      x : ↑(D.U i✝)
      y✝ : ↑(D.U j✝)
      h : Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Sigma.ι D.d …
      i j : D.J
      y : ↑(D.diagram.left { fst := i, snd := j })
      ⊢ D.Rel ⟨D.diagram.fstFrom { as := { fst := i, snd := j } }.as, (D.diagram.fst …
    -/
    exact Or.inr ⟨y, ⟨rfl, rfl⟩⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      D : TopCat.GlueData
      i j : D.J
      x : ↑(D.U i)
      y : ↑(D.U j)
      ⊢ D.Rel ⟨i, x⟩ ⟨j, y⟩ → Eq ((D.ι i) x) ((D.ι j) y)
    -/
  · rintro (⟨⟨⟩⟩ | ⟨z, e₁, e₂⟩)
      /-
        case mpr.inl.refl
        D : TopCat.GlueData
        i : D.J
        x : ↑(D.U i)
        ⊢ Eq ((D.ι i) x) ((D.ι i) x)
      -/
    · rfl
      /-
        🎉 no goals
      -/
    /-
      case mpr.inr.intro.intro
      D : TopCat.GlueData
      i j : D.J
      x : ↑(D.U i)
      y : ↑(D.U j)
      z : ↑(D.V { fst := ⟨i, x⟩.fst, snd := ⟨j, y⟩.fst })
      e₁ : Eq ((D.f ⟨i, x⟩.fst ⟨j, y⟩.fst) z) ⟨i, x⟩.snd
      e₂ : Eq ((D.f ⟨j, y⟩.fst ⟨i, x⟩.fst) ((D.t ⟨i, x⟩.fst ⟨j, y⟩.fst) z)) ⟨j, y⟩.snd
      ⊢ Eq ((D.ι i) x) ((D.ι j) y)
    -/
    dsimp only at *
    -- Porting note: there were `subst e₁` and `subst e₂`, instead of the `rw`
    /-
      case mpr.inr.intro.intro
      D : TopCat.GlueData
      i j : D.J
      x : ↑(D.U i)
      y : ↑(D.U j)
      z : ↑(D.V { fst := ⟨i, x⟩.fst, snd := ⟨j, y⟩.fst })
      e₁ : Eq ((D.f i j) z) x
      e₂ : Eq ((D.f j i) ((D.t i j) z)) y
      ⊢ Eq ((D.ι i) x) ((D.ι j) y)
    -/
    rw [← e₁, ← e₂] at *
    /-
      case mpr.inr.intro.intro
      D : TopCat.GlueData
      i j : D.J
      x : ↑(D.U i)
      y : ↑(D.U j)
      z : ↑(D.V { fst := ⟨i, x⟩.fst, snd := ⟨j, y⟩.fst })
      e₁ : Eq ((D.f i j) z) ((D.f i j) z)
      e₂ : Eq ((D.f j i) ((D.t i j) z)) ((D.f j i) ((D.t i j) z))
      ⊢ Eq ((D.ι i) ((D.f i j) z)) ((D.ι j) ((D.f j i) ((D.t i j) z)))
    -/
    rw [D.glue_condition_apply]
    /-
      🎉 no goals
    -/


theorem ι_injective (i : D.J) : Function.Injective (𝖣.ι i) := by
  /-
    D : TopCat.GlueData
    i : D.J
    ⊢ Function.Injective ⇑(D.ι i)
  -/
  intro x y h
  /-
    D : TopCat.GlueData
    i : D.J
    x y : ↑(D.U i)
    h : Eq ((D.ι i) x) ((D.ι i) y)
    ⊢ Eq x y
  -/
  rcases (D.ι_eq_iff_rel _ _ _ _).mp h with (⟨⟨⟩⟩ | ⟨_, e₁, e₂⟩)
    /-
      case inl.refl
      D : TopCat.GlueData
      i : D.J
      x : ↑(D.U i)
      h : Eq ((D.ι i) x) ((D.ι i) x)
      ⊢ Eq x x
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case inr.intro.intro
      D : TopCat.GlueData
      i : D.J
      x y : ↑(D.U i)
      h : Eq ((D.ι i) x) ((D.ι i) y)
      w✝ : ↑(D.V { fst := ⟨i, x⟩.fst, snd := ⟨i, y⟩.fst })
      e₁ : Eq ((D.f ⟨i, x⟩.fst ⟨i, y⟩.fst) w✝) ⟨i, x⟩.snd
      e₂ : Eq ((D.f ⟨i, y⟩.fst ⟨i, x⟩.fst) ((D.t ⟨i, x⟩.fst ⟨i, y⟩.fst) w✝)) ⟨i, y⟩. …
      ⊢ Eq x y
    -/
  · dsimp only at *
    -- Porting note: there were `cases e₁` and `cases e₂`, instead of the `rw`
    /-
      case inr.intro.intro
      D : TopCat.GlueData
      i : D.J
      x y : ↑(D.U i)
      h : Eq ((D.ι i) x) ((D.ι i) y)
      w✝ : ↑(D.V { fst := ⟨i, x⟩.fst, snd := ⟨i, y⟩.fst })
      e₁ : Eq ((D.f i i) w✝) x
      e₂ : Eq ((D.f i i) ((D.t i i) w✝)) y
      ⊢ Eq x y
    -/
    rw [← e₁, ← e₂]
    /-
      case inr.intro.intro
      D : TopCat.GlueData
      i : D.J
      x y : ↑(D.U i)
      h : Eq ((D.ι i) x) ((D.ι i) y)
      w✝ : ↑(D.V { fst := ⟨i, x⟩.fst, snd := ⟨i, y⟩.fst })
      e₁ : Eq ((D.f i i) w✝) x
      e₂ : Eq ((D.f i i) ((D.t i i) w✝)) y
      ⊢ Eq ((D.f i i) w✝) ((D.f i i) ((D.t i i) w✝))
    -/
    simp
    /-
      🎉 no goals
    -/


instance ι_mono (i : D.J) : Mono (𝖣.ι i) :=
  (TopCat.mono_iff_injective _).mpr (D.ι_injective _)


theorem image_inter (i j : D.J) :
    Set.range (𝖣.ι i) ∩ Set.range (𝖣.ι j) = Set.range (D.f i j ≫ 𝖣.ι _) := by
  /-
    D : TopCat.GlueData
    i j : D.J
    ⊢ Eq (Inter.inter (Set.range ⇑(D.ι i)) (Set.range ⇑(D.ι j))) (Set.range ⇑(Cate …
  -/
  ext x
  /-
    case h
    D : TopCat.GlueData
    i j : D.J
    x : ↑D.glued
    ⊢ Iff (Membership.mem (Inter.inter (Set.range ⇑(D.ι i)) (Set.range ⇑(D.ι j)))  …
  -/
  constructor
    /-
      case h.mp
      D : TopCat.GlueData
      i j : D.J
      x : ↑D.glued
      ⊢ Membership.mem (Inter.inter (Set.range ⇑(D.ι i)) (Set.range ⇑(D.ι j))) x → M …
    -/
  · rintro ⟨⟨x₁, eq₁⟩, ⟨x₂, eq₂⟩⟩
    /-
      case h.mp.intro.intro.intro
      D : TopCat.GlueData
      i j : D.J
      x : ↑D.glued
      x₁ : ↑(D.U i)
      eq₁ : Eq ((D.ι i) x₁) x
      x₂ : ↑(D.U j)
      eq₂ : Eq ((D.ι j) x₂) x
      ⊢ Membership.mem (Set.range ⇑(CategoryTheory.CategoryStruct.comp (D.f i j) (D. …
    -/
    obtain ⟨⟨⟩⟩ | ⟨y, e₁, -⟩ := (D.ι_eq_iff_rel _ _ _ _).mp (eq₁.trans eq₂.symm)
    · exact ⟨inv (D.f i i) x₁, by
        -- Porting note (https://github.com/leanprover-community/mathlib4/issues/10745): was `simp [eq₁]`
        -- See https://github.com/leanprover-community/mathlib4/issues/5026
        rw [TopCat.comp_app, CategoryTheory.IsIso.inv_hom_id_apply, eq₁]⟩
    · -- Porting note: was
      -- dsimp only at *; substs e₁ eq₁; exact ⟨y, by simp⟩
      /-
        case h.mp.intro.intro.intro.inr.intro.intro
        D : TopCat.GlueData
        i j : D.J
        x : ↑D.glued
        x₁ : ↑(D.U i)
        eq₁ : Eq ((D.ι i) x₁) x
        x₂ : ↑(D.U j)
        eq₂ : Eq ((D.ι j) x₂) x
        y : ↑(D.V { fst := ⟨i, x₁⟩.fst, snd := ⟨j, x₂⟩.fst })
        e₁ : Eq ((D.f ⟨i, x₁⟩.fst ⟨j, x₂⟩.fst) y) ⟨i, x₁⟩.snd
        ⊢ Membership.mem (Set.range ⇑(CategoryTheory.CategoryStruct.comp (D.f i j) (D. …
      -/
      dsimp only at *
      /-
        case h.mp.intro.intro.intro.inr.intro.intro
        D : TopCat.GlueData
        i j : D.J
        x : ↑D.glued
        x₁ : ↑(D.U i)
        eq₁ : Eq ((D.ι i) x₁) x
        x₂ : ↑(D.U j)
        eq₂ : Eq ((D.ι j) x₂) x
        y : ↑(D.V { fst := ⟨i, x₁⟩.fst, snd := ⟨j, x₂⟩.fst })
        e₁ : Eq ((D.f i j) y) x₁
        ⊢ Membership.mem (Set.range ⇑(CategoryTheory.CategoryStruct.comp (D.f i j) (D. …
      -/
      substs eq₁
      /-
        case h.mp.intro.intro.intro.inr.intro.intro
        D : TopCat.GlueData
        i j : D.J
        x₁ : ↑(D.U i)
        x₂ : ↑(D.U j)
        y : ↑(D.V { fst := ⟨i, x₁⟩.fst, snd := ⟨j, x₂⟩.fst })
        e₁ : Eq ((D.f i j) y) x₁
        eq₂ : Eq ((D.ι j) x₂) ((D.ι i) x₁)
        ⊢ Membership.mem (Set.range ⇑(CategoryTheory.CategoryStruct.comp (D.f i j) (D. …
      -/
      exact ⟨y, by simp [e₁]⟩
      /-
        🎉 no goals
      -/
    /-
      case h.mpr
      D : TopCat.GlueData
      i j : D.J
      x : ↑D.glued
      ⊢ Membership.mem (Set.range ⇑(CategoryTheory.CategoryStruct.comp (D.f i j) (D. …
    -/
  · rintro ⟨x, hx⟩
    /-
      case h.mpr.intro
      D : TopCat.GlueData
      i j : D.J
      x✝ : ↑D.glued
      x : ↑(D.V { fst := i, snd := j })
      hx : Eq ((CategoryTheory.CategoryStruct.comp (D.f i j) (D.ι i)) x) x✝
      ⊢ Membership.mem (Inter.inter (Set.range ⇑(D.ι i)) (Set.range ⇑(D.ι j))) x✝
    -/
    refine ⟨⟨D.f i j x, hx⟩, ⟨D.f j i (D.t _ _ x), ?_⟩⟩
    /-
      case h.mpr.intro
      D : TopCat.GlueData
      i j : D.J
      x✝ : ↑D.glued
      x : ↑(D.V { fst := i, snd := j })
      hx : Eq ((CategoryTheory.CategoryStruct.comp (D.f i j) (D.ι i)) x) x✝
      ⊢ Eq ((D.ι j) ((D.f j i) ((D.t i j) x))) x✝
    -/
    rw [D.glue_condition_apply]
    /-
      case h.mpr.intro
      D : TopCat.GlueData
      i j : D.J
      x✝ : ↑D.glued
      x : ↑(D.V { fst := i, snd := j })
      hx : Eq ((CategoryTheory.CategoryStruct.comp (D.f i j) (D.ι i)) x) x✝
      ⊢ Eq ((D.ι i) ((D.f i j) x)) x✝
    -/
    exact hx
    /-
      🎉 no goals
    -/


theorem preimage_range (i j : D.J) : 𝖣.ι j ⁻¹' Set.range (𝖣.ι i) = Set.range (D.f j i) := by
  rw [← Set.preimage_image_eq (Set.range (D.f j i)) (D.ι_injective j), ← Set.image_univ, ←
    Set.image_univ, ← Set.image_comp, ← coe_comp, Set.image_univ, Set.image_univ, ← image_inter,
    Set.preimage_range_inter]


theorem preimage_image_eq_image (i j : D.J) (U : Set (𝖣.U i)) :
    𝖣.ι j ⁻¹' (𝖣.ι i '' U) = D.f _ _ '' ((D.t j i ≫ D.f _ _) ⁻¹' U) := by
  have : D.f _ _ ⁻¹' (𝖣.ι j ⁻¹' (𝖣.ι i '' U)) = (D.t j i ≫ D.f _ _) ⁻¹' U := by
    ext x
    conv_rhs => rw [← Set.preimage_image_eq U (D.ι_injective _)]
    generalize 𝖣.ι i '' U = U' -- next 4 lines were `simp` before https://github.com/leanprover-community/mathlib4/pull/13170
    simp only [GlueData.diagram_l, GlueData.diagram_r, Set.mem_preimage, coe_comp,
      Function.comp_apply]
    rw [D.glue_condition_apply]
  /-
    D : TopCat.GlueData
    i j : D.J
    U : Set ↑(D.U i)
    this : Eq (Set.preimage (⇑(D.f j i)) (Set.preimage (⇑(D.ι j)) (Set.image (⇑(D. …
    ⊢ Eq (Set.preimage (⇑(D.ι j)) (Set.image (⇑(D.ι i)) U)) (Set.image (⇑(D.f j i) …
  -/
  rw [← this, Set.image_preimage_eq_inter_range]
  /-
    D : TopCat.GlueData
    i j : D.J
    U : Set ↑(D.U i)
    this : Eq (Set.preimage (⇑(D.f j i)) (Set.preimage (⇑(D.ι j)) (Set.image (⇑(D. …
    ⊢ Eq (Set.preimage (⇑(D.ι j)) (Set.image (⇑(D.ι i)) U)) (Inter.inter (Set.prei …
  -/
  symm
  /-
    D : TopCat.GlueData
    i j : D.J
    U : Set ↑(D.U i)
    this : Eq (Set.preimage (⇑(D.f j i)) (Set.preimage (⇑(D.ι j)) (Set.image (⇑(D. …
    ⊢ Eq (Inter.inter (Set.preimage (⇑(D.ι j)) (Set.image (⇑(D.ι i)) U)) (Set.rang …
  -/
  apply Set.inter_eq_self_of_subset_left
  /-
    case a
    D : TopCat.GlueData
    i j : D.J
    U : Set ↑(D.U i)
    this : Eq (Set.preimage (⇑(D.f j i)) (Set.preimage (⇑(D.ι j)) (Set.image (⇑(D. …
    ⊢ HasSubset.Subset (Set.preimage (⇑(D.ι j)) (Set.image (⇑(D.ι i)) U)) (Set.ran …
  -/
  rw [← D.preimage_range i j]
  /-
    case a
    D : TopCat.GlueData
    i j : D.J
    U : Set ↑(D.U i)
    this : Eq (Set.preimage (⇑(D.f j i)) (Set.preimage (⇑(D.ι j)) (Set.image (⇑(D. …
    ⊢ HasSubset.Subset (Set.preimage (⇑(D.ι j)) (Set.image (⇑(D.ι i)) U)) (Set.pre …
  -/
  exact Set.preimage_mono (Set.image_subset_range _ _)
  /-
    🎉 no goals
  -/


theorem preimage_image_eq_image' (i j : D.J) (U : Set (𝖣.U i)) :
    𝖣.ι j ⁻¹' (𝖣.ι i '' U) = (D.t i j ≫ D.f _ _) '' (D.f _ _ ⁻¹' U) := by
  /-
    D : TopCat.GlueData
    i j : D.J
    U : Set ↑(D.U i)
    ⊢ Eq (Set.preimage (⇑(D.ι j)) (Set.image (⇑(D.ι i)) U)) (Set.image (⇑(Category …
  -/
  convert D.preimage_image_eq_image i j U using 1
  /-
    case h.e'_3
    D : TopCat.GlueData
    i j : D.J
    U : Set ↑(D.U i)
    ⊢ Eq (Set.image (⇑(CategoryTheory.CategoryStruct.comp (D.t i j) (D.f j i))) (S …
  -/
  rw [coe_comp, coe_comp]
  -- Porting note: `show` was not needed, since `rw [← Set.image_image]` worked.
  /-
    case h.e'_3
    D : TopCat.GlueData
    i j : D.J
    U : Set ↑(D.U i)
    ⊢ Eq (Set.image (Function.comp ⇑(D.f j i) ⇑(D.t i j)) (Set.preimage (⇑(D.f i j …
  -/
  show (fun x => ((forget TopCat).map _ ((forget TopCat).map _ x))) '' _ = _
  /-
    case h.e'_3
    D : TopCat.GlueData
    i j : D.J
    U : Set ↑(D.U i)
    ⊢ Eq (Set.image (fun x => (CategoryTheory.forget TopCat).map (D.f j i) ((Categ …
  -/
  rw [← Set.image_image]
  /-
    case h.e'_3
    D : TopCat.GlueData
    i j : D.J
    U : Set ↑(D.U i)
    ⊢ Eq (Set.image ((CategoryTheory.forget TopCat).map (D.f j i)) (Set.image ((Ca …
  -/
  congr! 1
  /-
    case h.e'_3.h.e'_4
    D : TopCat.GlueData
    i j : D.J
    U : Set ↑(D.U i)
    ⊢ Eq (Set.image ((CategoryTheory.forget TopCat).map (D.t i j)) (Set.preimage ( …
  -/
  rw [← Set.eq_preimage_iff_image_eq, Set.preimage_preimage]
    /-
      case h.e'_3.h.e'_4
      D : TopCat.GlueData
      i j : D.J
      U : Set ↑(D.U i)
      ⊢ Eq (Set.preimage (⇑(D.f i j)) U) (Set.preimage (fun x => Function.comp (⇑(D. …
    -/
  · change _ = (D.t i j ≫ D.t j i ≫ _) ⁻¹' _
    /-
      case h.e'_3.h.e'_4
      D : TopCat.GlueData
      i j : D.J
      U : Set ↑(D.U i)
      ⊢ Eq (Set.preimage (⇑(D.f i j)) U) (Set.preimage (⇑(CategoryTheory.CategoryStr …
    -/
    rw [𝖣.t_inv_assoc]
    /-
      🎉 no goals
    -/
  /-
    case h.e'_3.h.e'_4.hf
    D : TopCat.GlueData
    i j : D.J
    U : Set ↑(D.U i)
    ⊢ Function.Bijective ((CategoryTheory.forget TopCat).map (D.t i j))
  -/
  rw [← isIso_iff_bijective]
  /-
    case h.e'_3.h.e'_4.hf
    D : TopCat.GlueData
    i j : D.J
    U : Set ↑(D.U i)
    ⊢ CategoryTheory.IsIso ((CategoryTheory.forget TopCat).map (D.t i j))
  -/
  apply (forget TopCat).map_isIso
  /-
    🎉 no goals
  -/


theorem open_image_open (i : D.J) (U : Opens (𝖣.U i)) : IsOpen (𝖣.ι i '' U) := by
  /-
    D : TopCat.GlueData
    i : D.J
    U : TopologicalSpace.Opens ↑(D.U i)
    ⊢ IsOpen (Set.image ⇑(D.ι i) ↑U)
  -/
  rw [isOpen_iff]
  /-
    D : TopCat.GlueData
    i : D.J
    U : TopologicalSpace.Opens ↑(D.U i)
    ⊢ ∀ (i_1 : D.J), IsOpen (Set.preimage (⇑(D.ι i_1)) (Set.image ⇑(D.ι i) ↑U))
  -/
  intro j
  /-
    D : TopCat.GlueData
    i : D.J
    U : TopologicalSpace.Opens ↑(D.U i)
    j : D.J
    ⊢ IsOpen (Set.preimage (⇑(D.ι j)) (Set.image ⇑(D.ι i) ↑U))
  -/
  rw [preimage_image_eq_image]
  /-
    D : TopCat.GlueData
    i : D.J
    U : TopologicalSpace.Opens ↑(D.U i)
    j : D.J
    ⊢ IsOpen (Set.image (⇑(D.f j i)) (Set.preimage ⇑(CategoryTheory.CategoryStruct …
  -/
  apply (D.f_open _ _).isOpenMap
  /-
    case a
    D : TopCat.GlueData
    i : D.J
    U : TopologicalSpace.Opens ↑(D.U i)
    j : D.J
    ⊢ IsOpen (Set.preimage ⇑(CategoryTheory.CategoryStruct.comp (D.t j i) (D.f i j …
  -/
  apply (D.t j i ≫ D.f i j).continuous_toFun.isOpen_preimage
  /-
    case a.a
    D : TopCat.GlueData
    i : D.J
    U : TopologicalSpace.Opens ↑(D.U i)
    j : D.J
    ⊢ IsOpen ↑U
  -/
  exact U.isOpen
  /-
    🎉 no goals
  -/


theorem ι_isOpenEmbedding (i : D.J) : IsOpenEmbedding (𝖣.ι i) :=
  .of_continuous_injective_isOpenMap (𝖣.ι i).continuous_toFun (D.ι_injective i) fun U h =>
    D.open_image_open i ⟨U, h⟩


@[deprecated (since := "2024-10-18")]
alias ι_openEmbedding := ι_isOpenEmbedding


/-- A family of gluing data consists of
1. An index type `J`
2. A bundled topological space `U i` for each `i : J`.
3. An open set `V i j ⊆ U i` for each `i j : J`.
4. A transition map `t i j : V i j ⟶ V j i` for each `i j : ι`.
such that
6. `V i i = U i`.
7. `t i i` is the identity.
8. For each `x ∈ V i j ∩ V i k`, `t i j x ∈ V j k`.
9. `t j k (t i j x) = t i k x`.

We can then glue the topological spaces `U i` together by identifying `V i j` with `V j i`.
-/
-- Porting note (https://github.com/leanprover-community/mathlib4/issues/5171): removed `@[nolint has_nonempty_instance]`
structure MkCore where
  {J : Type u}
  U : J → TopCat.{u}
  V : ∀ i, J → Opens (U i)
  t : ∀ i j, (Opens.toTopCat _).obj (V i j) ⟶ (Opens.toTopCat _).obj (V j i)
  V_id : ∀ i, V i i = ⊤
  t_id : ∀ i, ⇑(t i i) = id
  t_inter : ∀ ⦃i j⦄ (k) (x : V i j), ↑x ∈ V i k → (((↑) : (V j i) → (U j)) (t i j x)) ∈ V j k
  cocycle :
    ∀ (i j k) (x : V i j) (h : ↑x ∈ V i k),
      -- Porting note: the underscore in the next line was `↑(t i j x)`, but Lean type-mismatched
      (((↑) : (V k j) → (U k)) (t j k ⟨_, t_inter k x h⟩)) = ((↑) : (V k i) → (U k)) (t i k ⟨x, h⟩)


theorem MkCore.t_inv (h : MkCore) (i j : h.J) (x : h.V j i) : h.t i j ((h.t j i) x) = x := by
  /-
    h : TopCat.GlueData.MkCore
    i j : h.J
    x : Subtype fun x => Membership.mem (h.V j i) x
    ⊢ Eq ((h.t i j) ((h.t j i) x)) x
  -/
  have := h.cocycle j i j x ?_
    /-
      case refine_2
      h : TopCat.GlueData.MkCore
      i j : h.J
      x : Subtype fun x => Membership.mem (h.V j i) x
      this : Eq ↑((h.t i j) ⟨↑((h.t j i) x), ⋯⟩) ↑((h.t j j) ⟨↑x, ?refine_1⟩)
      ⊢ Eq ((h.t i j) ((h.t j i) x)) x
    -/
  · rw [h.t_id] at this
      /-
        case refine_2
        h : TopCat.GlueData.MkCore
        i j : h.J
        x : Subtype fun x => Membership.mem (h.V j i) x
        this : Eq ↑((h.t i j) ⟨↑((h.t j i) x), ⋯⟩) ↑(id ⟨↑x, ?refine_1⟩)
        ⊢ Eq ((h.t i j) ((h.t j i) x)) x
      -/
    · convert Subtype.eq this
      /-
        🎉 no goals
      -/
  /-
    case refine_1
    h : TopCat.GlueData.MkCore
    i j : h.J
    x : Subtype fun x => Membership.mem (h.V j i) x
    ⊢ Membership.mem (h.V j j) ↑x
  -/
  rw [h.V_id]
  /-
    case refine_1
    h : TopCat.GlueData.MkCore
    i j : h.J
    x : Subtype fun x => Membership.mem (h.V j i) x
    ⊢ Membership.mem Top.top ↑x
  -/
  trivial
  /-
    🎉 no goals
  -/


instance (h : MkCore.{u}) (i j : h.J) : IsIso (h.t i j) := by
  /-
    D : TopCat.GlueData
    h : TopCat.GlueData.MkCore
    i j : h.J
    ⊢ CategoryTheory.IsIso (h.t i j)
  -/
  use h.t j i; constructor <;> ext1; exacts [h.t_inv _ _ _, h.t_inv _ _ _]
                                     /-
                                       🎉 no goals
                                     -/


/-- (Implementation) the restricted transition map to be fed into `TopCat.GlueData`. -/
def MkCore.t' (h : MkCore.{u}) (i j k : h.J) :
    pullback (h.V i j).inclusion' (h.V i k).inclusion' ⟶
      pullback (h.V j k).inclusion' (h.V j i).inclusion' := by
  /-
    D : TopCat.GlueData
    h : TopCat.GlueData.MkCore
    i j k : h.J
    ⊢ Quiver.Hom (CategoryTheory.Limits.pullback (h.V i j).inclusion' (h.V i k).in …
  -/
  refine (pullbackIsoProdSubtype _ _).hom ≫ ⟨?_, ?_⟩ ≫ (pullbackIsoProdSubtype _ _).inv
    /-
      case refine_1
      D : TopCat.GlueData
      h : TopCat.GlueData.MkCore
      i j k : h.J
      ⊢ ↑(TopCat.of (Subtype fun p => Eq ((h.V i j).inclusion' p.1) ((h.V i k).inclu …
    -/
  · intro x
    /-
      case refine_1
      D : TopCat.GlueData
      h : TopCat.GlueData.MkCore
      i j k : h.J
      x : ↑(TopCat.of (Subtype fun p => Eq ((h.V i j).inclusion' p.1) ((h.V i k).inc …
      ⊢ ↑(TopCat.of (Subtype fun p => Eq ((h.V j k).inclusion' p.1) ((h.V j i).inclu …
    -/
    refine ⟨⟨⟨(h.t i j x.1.1).1, ?_⟩, h.t i j x.1.1⟩, rfl⟩
    /-
      case refine_1
      D : TopCat.GlueData
      h : TopCat.GlueData.MkCore
      i j k : h.J
      x : ↑(TopCat.of (Subtype fun p => Eq ((h.V i j).inclusion' p.1) ((h.V i k).inc …
      ⊢ Membership.mem (h.V j k) ↑((h.t i j) (↑x).1)
    -/
    rcases x with ⟨⟨⟨x, hx⟩, ⟨x', hx'⟩⟩, rfl : x = x'⟩
    /-
      case refine_1.mk.mk.mk.mk
      D : TopCat.GlueData
      h : TopCat.GlueData.MkCore
      i j k : h.J
      x : ↑(h.U i)
      hx : Membership.mem (h.V i j) x
      hx' : Membership.mem (h.V i k) x
      ⊢ Membership.mem (h.V j k) ↑((h.t i j) (↑⟨{ fst := ⟨x, hx⟩, snd := ⟨x, hx'⟩ }, …
    -/
    exact h.t_inter _ ⟨x, hx⟩ hx'
    /-
      🎉 no goals
    -/
  /-
    case refine_2
    D : TopCat.GlueData
    h : TopCat.GlueData.MkCore
    i j k : h.J
    ⊢ Continuous fun x => ⟨{ fst := ⟨↑((h.t i j) (↑x).1), ⋯⟩, snd := (h.t i j) (↑x …
  -/
  fun_prop
  /-
    🎉 no goals
  -/


/-- This is a constructor of `TopCat.GlueData` whose arguments are in terms of elements and
intersections rather than subobjects and pullbacks. Please refer to `TopCat.GlueData.MkCore` for
details. -/
def mk' (h : MkCore.{u}) : TopCat.GlueData where
  J := h.J
  U := h.U
  V i := (Opens.toTopCat _).obj (h.V i.1 i.2)
  f i j := (h.V i j).inclusion'
  f_id i := by
    -- Porting note (https://github.com/leanprover-community/mathlib4/issues/12129): additional beta reduction needed
    /-
      D : TopCat.GlueData
      h : TopCat.GlueData.MkCore
      i : h.J
      ⊢ CategoryTheory.IsIso ((fun i j => (h.V i j).inclusion') i i)
    -/
    beta_reduce
    /-
      D : TopCat.GlueData
      h : TopCat.GlueData.MkCore
      i : h.J
      ⊢ CategoryTheory.IsIso (h.V i i).inclusion'
    -/
    exact (h.V_id i).symm ▸ (Opens.inclusionTopIso (h.U i)).isIso_hom
    /-
      🎉 no goals
    -/
  f_open := fun i j : h.J => (h.V i j).isOpenEmbedding
  t := h.t
               /-
                 D : TopCat.GlueData
                 h : TopCat.GlueData.MkCore
                 i : h.J
                 ⊢ Eq (h.t i i) (CategoryTheory.CategoryStruct.id ((fun i => (TopologicalSpace. …
               -/
  t_id i := by ext; rw [h.t_id]; rfl
                                 /-
                                   🎉 no goals
                                 -/
  t' := h.t'
  t_fac i j k := by
    /-
      D : TopCat.GlueData
      h : TopCat.GlueData.MkCore
      i j k : h.J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (h.t' i j k) (CategoryTheory.Limits.p …
    -/
    delta MkCore.t'
    rw [Category.assoc, Category.assoc, pullbackIsoProdSubtype_inv_snd, ← Iso.eq_inv_comp,
      pullbackIsoProdSubtype_inv_fst_assoc]
    /-
      D : TopCat.GlueData
      h : TopCat.GlueData.MkCore
      i j k : h.J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp { toFun := fun x => ⟨{ fst := ⟨↑((h.t …
    -/
    ext ⟨⟨⟨x, hx⟩, ⟨x', hx'⟩⟩, rfl : x = x'⟩
    /-
      case w.mk.mk.mk.mk
      D : TopCat.GlueData
      h : TopCat.GlueData.MkCore
      i j k : h.J
      x : ↑(h.U i)
      hx : Membership.mem (h.V i j) x
      hx' : Membership.mem (h.V i k) x
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp { toFun := fun x => ⟨{ fst := ⟨↑((h. …
    -/
    rfl
    /-
      🎉 no goals
    -/
  cocycle i j k := by
    /-
      D : TopCat.GlueData
      h : TopCat.GlueData.MkCore
      i j k : h.J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (h.t' i j k) (CategoryTheory.Category …
    -/
    delta MkCore.t'
    /-
      D : TopCat.GlueData
      h : TopCat.GlueData.MkCore
      i j k : h.J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    simp_rw [← Category.assoc]
    /-
      D : TopCat.GlueData
      h : TopCat.GlueData.MkCore
      i j k : h.J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    rw [Iso.comp_inv_eq]
    /-
      D : TopCat.GlueData
      h : TopCat.GlueData.MkCore
      i j k : h.J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    simp only [Iso.inv_hom_id_assoc, Category.assoc, Category.id_comp]
    /-
      D : TopCat.GlueData
      h : TopCat.GlueData.MkCore
      i j k : h.J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (TopCat.pullbackIsoProdSubtype (h.V i …
    -/
    rw [← Iso.eq_inv_comp, Iso.inv_hom_id]
    /-
      D : TopCat.GlueData
      h : TopCat.GlueData.MkCore
      i j k : h.J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp { toFun := fun x => ⟨{ fst := ⟨↑((h.t …
    -/
    ext1 ⟨⟨⟨x, hx⟩, ⟨x', hx'⟩⟩, rfl : x = x'⟩
    rw [comp_app, ContinuousMap.coe_mk, comp_app, id_app, ContinuousMap.coe_mk, Subtype.mk_eq_mk,
      Prod.mk.inj_iff, Subtype.mk_eq_mk, Subtype.ext_iff, and_self_iff]
    /-
      case w.mk.mk.mk.mk
      D : TopCat.GlueData
      h : TopCat.GlueData.MkCore
      i j k : h.J
      x : ↑(h.U i)
      hx : Membership.mem (h.V i j) x
      hx' : Membership.mem (h.V i k) x
      ⊢ Eq (↑((h.t k i) (↑({ toFun := fun x => ⟨{ fst := ⟨↑((h.t j k) (↑x).1), ⋯⟩, s …
    -/
    convert congr_arg Subtype.val (h.t_inv k i ⟨x, hx'⟩) using 3
    /-
      case h.e'_2.h.e'_3.h.e'_6
      D : TopCat.GlueData
      h : TopCat.GlueData.MkCore
      i j k : h.J
      x : ↑(h.U i)
      hx : Membership.mem (h.V i j) x
      hx' : Membership.mem (h.V i k) x
      ⊢ Eq (↑({ toFun := fun x => ⟨{ fst := ⟨↑((h.t j k) (↑x).1), ⋯⟩, snd := (h.t j  …
    -/
    refine Subtype.ext ?_
    /-
      case h.e'_2.h.e'_3.h.e'_6
      D : TopCat.GlueData
      h : TopCat.GlueData.MkCore
      i j k : h.J
      x : ↑(h.U i)
      hx : Membership.mem (h.V i j) x
      hx' : Membership.mem (h.V i k) x
      ⊢ Eq ↑(↑({ toFun := fun x => ⟨{ fst := ⟨↑((h.t j k) (↑x).1), ⋯⟩, snd := (h.t j …
    -/
    exact h.cocycle i j k ⟨x, hx⟩ hx'
    /-
      🎉 no goals
    -/
  -- Porting note: was not necessary in mathlib3
  f_mono _ _ := (TopCat.mono_iff_injective _).mpr fun _ _ h => Subtype.ext h


/-- We may construct a glue data from a family of open sets. -/
@[simps! toGlueData_J toGlueData_U toGlueData_V toGlueData_t toGlueData_f]
def ofOpenSubsets : TopCat.GlueData.{u} :=
  mk'.{u}
    { J
      U := fun i => (Opens.toTopCat <| TopCat.of α).obj (U i)
      V := fun _ j => (Opens.map <| Opens.inclusion' _).obj (U j)
                                                          /-
                                                            D : TopCat.GlueData
                                                            α : Type u
                                                            inst✝ : TopologicalSpace α
                                                            J : Type u
                                                            U : J → TopologicalSpace.Opens α
                                                            i j : J
                                                            ⊢ Continuous fun x => ⟨⟨↑↑x, ⋯⟩, ⋯⟩
                                                          -/
      t := fun i j => ⟨fun x => ⟨⟨x.1.1, x.2⟩, x.1.2⟩, by fun_prop⟩
                                                          /-
                                                            🎉 no goals
                                                          -/
                          /-
                            D : TopCat.GlueData
                            α : Type u
                            inst✝ : TopologicalSpace α
                            J : Type u
                            U : J → TopologicalSpace.Opens α
                            i : J
                            ⊢ Eq ((fun x j => (TopologicalSpace.Opens.map (U x).inclusion').obj (U j)) i i …
                          -/
      V_id := fun i => by ext; simp
                               /-
                                 🎉 no goals
                               -/
                          /-
                            D : TopCat.GlueData
                            α : Type u
                            inst✝ : TopologicalSpace α
                            J : Type u
                            U : J → TopologicalSpace.Opens α
                            i : J
                            ⊢ Eq (⇑((fun i j => { toFun := fun x => ⟨⟨↑↑x, ⋯⟩, ⋯⟩, continuous_toFun := ⋯ } …
                          -/
      t_id := fun i => by ext; rfl
                               /-
                                 🎉 no goals
                               -/
      t_inter := fun _ _ _ _ hx => hx
      cocycle := fun _ _ _ _ _ => rfl }


/-- The canonical map from the glue of a family of open subsets `α` into `α`.
This map is an open embedding (`fromOpenSubsetsGlue_isOpenEmbedding`),
and its range is `⋃ i, (U i : Set α)` (`range_fromOpenSubsetsGlue`).
-/
def fromOpenSubsetsGlue : (ofOpenSubsets U).toGlueData.glued ⟶ TopCat.of α :=
                                                              /-
                                                                D : TopCat.GlueData
                                                                α : Type u
                                                                inst✝ : TopologicalSpace α
                                                                J : Type u
                                                                U : J → TopologicalSpace.Opens α
                                                                ⊢ ∀ (a : (TopCat.GlueData.ofOpenSubsets U).diagram.L), Eq (CategoryTheory.Cate …
                                                              -/
  Multicoequalizer.desc _ _ (fun _ => Opens.inclusion' _) (by rintro ⟨i, j⟩; ext x; rfl)
                                                                                    /-
                                                                                      🎉 no goals
                                                                                    -/

-- Porting note: `elementwise` here produces a bad lemma,
-- where too much has been simplified, despite the `nosimp`.

@[simp, elementwise nosimp]
theorem ι_fromOpenSubsetsGlue (i : J) :
    (ofOpenSubsets U).toGlueData.ι i ≫ fromOpenSubsetsGlue U = Opens.inclusion' _ :=
  Multicoequalizer.π_desc _ _ _ _ _


theorem fromOpenSubsetsGlue_injective : Function.Injective (fromOpenSubsetsGlue U) := by
  /-
    α : Type u
    inst✝ : TopologicalSpace α
    J : Type u
    U : J → TopologicalSpace.Opens α
    ⊢ Function.Injective ⇑(TopCat.GlueData.fromOpenSubsetsGlue U)
  -/
  intro x y e
  /-
    α : Type u
    inst✝ : TopologicalSpace α
    J : Type u
    U : J → TopologicalSpace.Opens α
    x y : ↑(TopCat.GlueData.ofOpenSubsets U).glued
    e : Eq ((TopCat.GlueData.fromOpenSubsetsGlue U) x) ((TopCat.GlueData.fromOpenS …
    ⊢ Eq x y
  -/
  obtain ⟨i, ⟨x, hx⟩, rfl⟩ := (ofOpenSubsets U).ι_jointly_surjective x
  /-
    case intro.intro.mk
    α : Type u
    inst✝ : TopologicalSpace α
    J : Type u
    U : J → TopologicalSpace.Opens α
    y : ↑(TopCat.GlueData.ofOpenSubsets U).glued
    i : (TopCat.GlueData.ofOpenSubsets U).J
    x : ↑(TopCat.of α)
    hx : Membership.mem (U i) x
    e : Eq ((TopCat.GlueData.fromOpenSubsetsGlue U) (((TopCat.GlueData.ofOpenSubse …
    ⊢ Eq (((TopCat.GlueData.ofOpenSubsets U).ι i) ⟨x, hx⟩) y
  -/
  obtain ⟨j, ⟨y, hy⟩, rfl⟩ := (ofOpenSubsets U).ι_jointly_surjective y
  -- see the porting note on `ι_fromOpenSubsetsGlue`
  /-
    case intro.intro.mk.intro.intro.mk
    α : Type u
    inst✝ : TopologicalSpace α
    J : Type u
    U : J → TopologicalSpace.Opens α
    i : (TopCat.GlueData.ofOpenSubsets U).J
    x : ↑(TopCat.of α)
    hx : Membership.mem (U i) x
    j : (TopCat.GlueData.ofOpenSubsets U).J
    y : ↑(TopCat.of α)
    hy : Membership.mem (U j) y
    e : Eq ((TopCat.GlueData.fromOpenSubsetsGlue U) (((TopCat.GlueData.ofOpenSubse …
    ⊢ Eq (((TopCat.GlueData.ofOpenSubsets U).ι i) ⟨x, hx⟩) (((TopCat.GlueData.ofOp …
  -/
  rw [ι_fromOpenSubsetsGlue_apply, ι_fromOpenSubsetsGlue_apply] at e
  /-
    case intro.intro.mk.intro.intro.mk
    α : Type u
    inst✝ : TopologicalSpace α
    J : Type u
    U : J → TopologicalSpace.Opens α
    i : (TopCat.GlueData.ofOpenSubsets U).J
    x : ↑(TopCat.of α)
    hx : Membership.mem (U i) x
    j : (TopCat.GlueData.ofOpenSubsets U).J
    y : ↑(TopCat.of α)
    hy : Membership.mem (U j) y
    e : Eq ((U i).inclusion' ⟨x, hx⟩) ((U j).inclusion' ⟨y, hy⟩)
    ⊢ Eq (((TopCat.GlueData.ofOpenSubsets U).ι i) ⟨x, hx⟩) (((TopCat.GlueData.ofOp …
  -/
  change x = y at e
  /-
    case intro.intro.mk.intro.intro.mk
    α : Type u
    inst✝ : TopologicalSpace α
    J : Type u
    U : J → TopologicalSpace.Opens α
    i : (TopCat.GlueData.ofOpenSubsets U).J
    x : ↑(TopCat.of α)
    hx : Membership.mem (U i) x
    j : (TopCat.GlueData.ofOpenSubsets U).J
    y : ↑(TopCat.of α)
    hy : Membership.mem (U j) y
    e : Eq x y
    ⊢ Eq (((TopCat.GlueData.ofOpenSubsets U).ι i) ⟨x, hx⟩) (((TopCat.GlueData.ofOp …
  -/
  subst e
  /-
    case intro.intro.mk.intro.intro.mk
    α : Type u
    inst✝ : TopologicalSpace α
    J : Type u
    U : J → TopologicalSpace.Opens α
    i : (TopCat.GlueData.ofOpenSubsets U).J
    x : ↑(TopCat.of α)
    hx : Membership.mem (U i) x
    j : (TopCat.GlueData.ofOpenSubsets U).J
    hy : Membership.mem (U j) x
    ⊢ Eq (((TopCat.GlueData.ofOpenSubsets U).ι i) ⟨x, hx⟩) (((TopCat.GlueData.ofOp …
  -/
  rw [(ofOpenSubsets U).ι_eq_iff_rel]
  /-
    case intro.intro.mk.intro.intro.mk
    α : Type u
    inst✝ : TopologicalSpace α
    J : Type u
    U : J → TopologicalSpace.Opens α
    i : (TopCat.GlueData.ofOpenSubsets U).J
    x : ↑(TopCat.of α)
    hx : Membership.mem (U i) x
    j : (TopCat.GlueData.ofOpenSubsets U).J
    hy : Membership.mem (U j) x
    ⊢ (TopCat.GlueData.ofOpenSubsets U).Rel ⟨i, ⟨x, hx⟩⟩ ⟨j, ⟨x, hy⟩⟩
  -/
  right
  /-
    case intro.intro.mk.intro.intro.mk.h
    α : Type u
    inst✝ : TopologicalSpace α
    J : Type u
    U : J → TopologicalSpace.Opens α
    i : (TopCat.GlueData.ofOpenSubsets U).J
    x : ↑(TopCat.of α)
    hx : Membership.mem (U i) x
    j : (TopCat.GlueData.ofOpenSubsets U).J
    hy : Membership.mem (U j) x
    ⊢ Exists fun x_1 => And (Eq (((TopCat.GlueData.ofOpenSubsets U).f ⟨i, ⟨x, hx⟩⟩ …
  -/
  exact ⟨⟨⟨x, hx⟩, hy⟩, rfl, rfl⟩
  /-
    🎉 no goals
  -/


theorem fromOpenSubsetsGlue_isOpenMap : IsOpenMap (fromOpenSubsetsGlue U) := by
  /-
    α : Type u
    inst✝ : TopologicalSpace α
    J : Type u
    U : J → TopologicalSpace.Opens α
    ⊢ IsOpenMap ⇑(TopCat.GlueData.fromOpenSubsetsGlue U)
  -/
  intro s hs
  /-
    α : Type u
    inst✝ : TopologicalSpace α
    J : Type u
    U : J → TopologicalSpace.Opens α
    s : Set ↑(TopCat.GlueData.ofOpenSubsets U).glued
    hs : IsOpen s
    ⊢ IsOpen (Set.image (⇑(TopCat.GlueData.fromOpenSubsetsGlue U)) s)
  -/
  rw [(ofOpenSubsets U).isOpen_iff] at hs
  /-
    α : Type u
    inst✝ : TopologicalSpace α
    J : Type u
    U : J → TopologicalSpace.Opens α
    s : Set ↑(TopCat.GlueData.ofOpenSubsets U).glued
    hs : ∀ (i : (TopCat.GlueData.ofOpenSubsets U).J), IsOpen (Set.preimage (⇑((Top …
    ⊢ IsOpen (Set.image (⇑(TopCat.GlueData.fromOpenSubsetsGlue U)) s)
  -/
  rw [isOpen_iff_forall_mem_open]
  /-
    α : Type u
    inst✝ : TopologicalSpace α
    J : Type u
    U : J → TopologicalSpace.Opens α
    s : Set ↑(TopCat.GlueData.ofOpenSubsets U).glued
    hs : ∀ (i : (TopCat.GlueData.ofOpenSubsets U).J), IsOpen (Set.preimage (⇑((Top …
    ⊢ ∀ (x : ↑(TopCat.of α)), Membership.mem (Set.image (⇑(TopCat.GlueData.fromOpe …
  -/
  rintro _ ⟨x, hx, rfl⟩
  /-
    case intro.intro
    α : Type u
    inst✝ : TopologicalSpace α
    J : Type u
    U : J → TopologicalSpace.Opens α
    s : Set ↑(TopCat.GlueData.ofOpenSubsets U).glued
    hs : ∀ (i : (TopCat.GlueData.ofOpenSubsets U).J), IsOpen (Set.preimage (⇑((Top …
    x : ↑(TopCat.GlueData.ofOpenSubsets U).glued
    hx : Membership.mem s x
    ⊢ Exists fun t => And (HasSubset.Subset t (Set.image (⇑(TopCat.GlueData.fromOp …
  -/
  obtain ⟨i, ⟨x, hx'⟩, rfl⟩ := (ofOpenSubsets U).ι_jointly_surjective x
  /-
    case intro.intro.intro.intro.mk
    α : Type u
    inst✝ : TopologicalSpace α
    J : Type u
    U : J → TopologicalSpace.Opens α
    s : Set ↑(TopCat.GlueData.ofOpenSubsets U).glued
    hs : ∀ (i : (TopCat.GlueData.ofOpenSubsets U).J), IsOpen (Set.preimage (⇑((Top …
    i : (TopCat.GlueData.ofOpenSubsets U).J
    x : ↑(TopCat.of α)
    hx' : Membership.mem (U i) x
    hx : Membership.mem s (((TopCat.GlueData.ofOpenSubsets U).ι i) ⟨x, hx'⟩)
    ⊢ Exists fun t => And (HasSubset.Subset t (Set.image (⇑(TopCat.GlueData.fromOp …
  -/
  use fromOpenSubsetsGlue U '' s ∩ Set.range (@Opens.inclusion' (TopCat.of α) (U i))
  /-
    case h
    α : Type u
    inst✝ : TopologicalSpace α
    J : Type u
    U : J → TopologicalSpace.Opens α
    s : Set ↑(TopCat.GlueData.ofOpenSubsets U).glued
    hs : ∀ (i : (TopCat.GlueData.ofOpenSubsets U).J), IsOpen (Set.preimage (⇑((Top …
    i : (TopCat.GlueData.ofOpenSubsets U).J
    x : ↑(TopCat.of α)
    hx' : Membership.mem (U i) x
    hx : Membership.mem s (((TopCat.GlueData.ofOpenSubsets U).ι i) ⟨x, hx'⟩)
    ⊢ And (HasSubset.Subset (Inter.inter (Set.image (⇑(TopCat.GlueData.fromOpenSub …
  -/
  use Set.inter_subset_left
  /-
    case right
    α : Type u
    inst✝ : TopologicalSpace α
    J : Type u
    U : J → TopologicalSpace.Opens α
    s : Set ↑(TopCat.GlueData.ofOpenSubsets U).glued
    hs : ∀ (i : (TopCat.GlueData.ofOpenSubsets U).J), IsOpen (Set.preimage (⇑((Top …
    i : (TopCat.GlueData.ofOpenSubsets U).J
    x : ↑(TopCat.of α)
    hx' : Membership.mem (U i) x
    hx : Membership.mem s (((TopCat.GlueData.ofOpenSubsets U).ι i) ⟨x, hx'⟩)
    ⊢ And (IsOpen (Inter.inter (Set.image (⇑(TopCat.GlueData.fromOpenSubsetsGlue U …
  -/
  constructor
    /-
      case right.left
      α : Type u
      inst✝ : TopologicalSpace α
      J : Type u
      U : J → TopologicalSpace.Opens α
      s : Set ↑(TopCat.GlueData.ofOpenSubsets U).glued
      hs : ∀ (i : (TopCat.GlueData.ofOpenSubsets U).J), IsOpen (Set.preimage (⇑((Top …
      i : (TopCat.GlueData.ofOpenSubsets U).J
      x : ↑(TopCat.of α)
      hx' : Membership.mem (U i) x
      hx : Membership.mem s (((TopCat.GlueData.ofOpenSubsets U).ι i) ⟨x, hx'⟩)
      ⊢ IsOpen (Inter.inter (Set.image (⇑(TopCat.GlueData.fromOpenSubsetsGlue U)) s) …
    -/
  · rw [← Set.image_preimage_eq_inter_range]
    /-
      case right.left
      α : Type u
      inst✝ : TopologicalSpace α
      J : Type u
      U : J → TopologicalSpace.Opens α
      s : Set ↑(TopCat.GlueData.ofOpenSubsets U).glued
      hs : ∀ (i : (TopCat.GlueData.ofOpenSubsets U).J), IsOpen (Set.preimage (⇑((Top …
      i : (TopCat.GlueData.ofOpenSubsets U).J
      x : ↑(TopCat.of α)
      hx' : Membership.mem (U i) x
      hx : Membership.mem s (((TopCat.GlueData.ofOpenSubsets U).ι i) ⟨x, hx'⟩)
      ⊢ IsOpen (Set.image (⇑(U i).inclusion') (Set.preimage (⇑(U i).inclusion') (Set …
    -/
    apply (Opens.isOpenEmbedding (X := TopCat.of α) (U i)).isOpenMap
    /-
      case right.left.a
      α : Type u
      inst✝ : TopologicalSpace α
      J : Type u
      U : J → TopologicalSpace.Opens α
      s : Set ↑(TopCat.GlueData.ofOpenSubsets U).glued
      hs : ∀ (i : (TopCat.GlueData.ofOpenSubsets U).J), IsOpen (Set.preimage (⇑((Top …
      i : (TopCat.GlueData.ofOpenSubsets U).J
      x : ↑(TopCat.of α)
      hx' : Membership.mem (U i) x
      hx : Membership.mem s (((TopCat.GlueData.ofOpenSubsets U).ι i) ⟨x, hx'⟩)
      ⊢ IsOpen (Set.preimage (⇑(U i).inclusion') (Set.image (⇑(TopCat.GlueData.fromO …
    -/
    convert hs i using 1
    /-
      case h.e'_3.h
      α : Type u
      inst✝ : TopologicalSpace α
      J : Type u
      U : J → TopologicalSpace.Opens α
      s : Set ↑(TopCat.GlueData.ofOpenSubsets U).glued
      hs : ∀ (i : (TopCat.GlueData.ofOpenSubsets U).J), IsOpen (Set.preimage (⇑((Top …
      i : (TopCat.GlueData.ofOpenSubsets U).J
      x : ↑(TopCat.of α)
      hx' : Membership.mem (U i) x
      hx : Membership.mem s (((TopCat.GlueData.ofOpenSubsets U).ι i) ⟨x, hx'⟩)
      e_1✝ : Eq ↑((TopologicalSpace.Opens.toTopCat (TopCat.of α)).obj (U i)) ↑((TopC …
      ⊢ Eq (Set.preimage (⇑(U i).inclusion') (Set.image (⇑(TopCat.GlueData.fromOpenS …
    -/
    erw [← ι_fromOpenSubsetsGlue, coe_comp, Set.preimage_comp]
    /-
      case h.e'_3.h
      α : Type u
      inst✝ : TopologicalSpace α
      J : Type u
      U : J → TopologicalSpace.Opens α
      s : Set ↑(TopCat.GlueData.ofOpenSubsets U).glued
      hs : ∀ (i : (TopCat.GlueData.ofOpenSubsets U).J), IsOpen (Set.preimage (⇑((Top …
      i : (TopCat.GlueData.ofOpenSubsets U).J
      x : ↑(TopCat.of α)
      hx' : Membership.mem (U i) x
      hx : Membership.mem s (((TopCat.GlueData.ofOpenSubsets U).ι i) ⟨x, hx'⟩)
      e_1✝ : Eq ↑((TopologicalSpace.Opens.toTopCat (TopCat.of α)).obj (U i)) ↑((TopC …
      ⊢ Eq (Set.preimage (⇑((TopCat.GlueData.ofOpenSubsets U).ι i)) (Set.preimage (⇑ …
    -/
    congr! 1
    /-
      case h.e'_3.h.h.e'_4
      α : Type u
      inst✝ : TopologicalSpace α
      J : Type u
      U : J → TopologicalSpace.Opens α
      s : Set ↑(TopCat.GlueData.ofOpenSubsets U).glued
      hs : ∀ (i : (TopCat.GlueData.ofOpenSubsets U).J), IsOpen (Set.preimage (⇑((Top …
      i : (TopCat.GlueData.ofOpenSubsets U).J
      x : ↑(TopCat.of α)
      hx' : Membership.mem (U i) x
      hx : Membership.mem s (((TopCat.GlueData.ofOpenSubsets U).ι i) ⟨x, hx'⟩)
      e_1✝ : Eq ↑((TopologicalSpace.Opens.toTopCat (TopCat.of α)).obj (U i)) ↑((TopC …
      ⊢ Eq (Set.preimage (⇑(TopCat.GlueData.fromOpenSubsetsGlue U)) (Set.image (⇑(To …
    -/
    exact Set.preimage_image_eq _ (fromOpenSubsetsGlue_injective U)
    /-
      🎉 no goals
    -/
    /-
      case right.right
      α : Type u
      inst✝ : TopologicalSpace α
      J : Type u
      U : J → TopologicalSpace.Opens α
      s : Set ↑(TopCat.GlueData.ofOpenSubsets U).glued
      hs : ∀ (i : (TopCat.GlueData.ofOpenSubsets U).J), IsOpen (Set.preimage (⇑((Top …
      i : (TopCat.GlueData.ofOpenSubsets U).J
      x : ↑(TopCat.of α)
      hx' : Membership.mem (U i) x
      hx : Membership.mem s (((TopCat.GlueData.ofOpenSubsets U).ι i) ⟨x, hx'⟩)
      ⊢ Membership.mem (Inter.inter (Set.image (⇑(TopCat.GlueData.fromOpenSubsetsGlu …
    -/
  · refine ⟨Set.mem_image_of_mem _ hx, ?_⟩
    /-
      case right.right
      α : Type u
      inst✝ : TopologicalSpace α
      J : Type u
      U : J → TopologicalSpace.Opens α
      s : Set ↑(TopCat.GlueData.ofOpenSubsets U).glued
      hs : ∀ (i : (TopCat.GlueData.ofOpenSubsets U).J), IsOpen (Set.preimage (⇑((Top …
      i : (TopCat.GlueData.ofOpenSubsets U).J
      x : ↑(TopCat.of α)
      hx' : Membership.mem (U i) x
      hx : Membership.mem s (((TopCat.GlueData.ofOpenSubsets U).ι i) ⟨x, hx'⟩)
      ⊢ Membership.mem (Set.range ⇑(U i).inclusion') ((TopCat.GlueData.fromOpenSubse …
    -/
    rw [ι_fromOpenSubsetsGlue_apply]
    /-
      case right.right
      α : Type u
      inst✝ : TopologicalSpace α
      J : Type u
      U : J → TopologicalSpace.Opens α
      s : Set ↑(TopCat.GlueData.ofOpenSubsets U).glued
      hs : ∀ (i : (TopCat.GlueData.ofOpenSubsets U).J), IsOpen (Set.preimage (⇑((Top …
      i : (TopCat.GlueData.ofOpenSubsets U).J
      x : ↑(TopCat.of α)
      hx' : Membership.mem (U i) x
      hx : Membership.mem s (((TopCat.GlueData.ofOpenSubsets U).ι i) ⟨x, hx'⟩)
      ⊢ Membership.mem (Set.range ⇑(U i).inclusion') ((U i).inclusion' ⟨x, hx'⟩)
    -/
    exact Set.mem_range_self _
    /-
      🎉 no goals
    -/


theorem fromOpenSubsetsGlue_isOpenEmbedding : IsOpenEmbedding (fromOpenSubsetsGlue U) :=
  .of_continuous_injective_isOpenMap (ContinuousMap.continuous_toFun _)
    (fromOpenSubsetsGlue_injective U) (fromOpenSubsetsGlue_isOpenMap U)


@[deprecated (since := "2024-10-18")]
alias fromOpenSubsetsGlue_openEmbedding := fromOpenSubsetsGlue_isOpenEmbedding


theorem range_fromOpenSubsetsGlue : Set.range (fromOpenSubsetsGlue U) = ⋃ i, (U i : Set α) := by
  /-
    α : Type u
    inst✝ : TopologicalSpace α
    J : Type u
    U : J → TopologicalSpace.Opens α
    ⊢ Eq (Set.range ⇑(TopCat.GlueData.fromOpenSubsetsGlue U)) (Set.iUnion fun i => …
  -/
  ext
  /-
    case h
    α : Type u
    inst✝ : TopologicalSpace α
    J : Type u
    U : J → TopologicalSpace.Opens α
    x✝ : ↑(TopCat.of α)
    ⊢ Iff (Membership.mem (Set.range ⇑(TopCat.GlueData.fromOpenSubsetsGlue U)) x✝) …
  -/
  constructor
    /-
      case h.mp
      α : Type u
      inst✝ : TopologicalSpace α
      J : Type u
      U : J → TopologicalSpace.Opens α
      x✝ : ↑(TopCat.of α)
      ⊢ Membership.mem (Set.range ⇑(TopCat.GlueData.fromOpenSubsetsGlue U)) x✝ → Mem …
    -/
  · rintro ⟨x, rfl⟩
    /-
      case h.mp.intro
      α : Type u
      inst✝ : TopologicalSpace α
      J : Type u
      U : J → TopologicalSpace.Opens α
      x : ↑(TopCat.GlueData.ofOpenSubsets U).glued
      ⊢ Membership.mem (Set.iUnion fun i => ↑(U i)) ((TopCat.GlueData.fromOpenSubset …
    -/
    obtain ⟨i, ⟨x, hx'⟩, rfl⟩ := (ofOpenSubsets U).ι_jointly_surjective x
    /-
      case h.mp.intro.intro.intro.mk
      α : Type u
      inst✝ : TopologicalSpace α
      J : Type u
      U : J → TopologicalSpace.Opens α
      i : (TopCat.GlueData.ofOpenSubsets U).J
      x : ↑(TopCat.of α)
      hx' : Membership.mem (U i) x
      ⊢ Membership.mem (Set.iUnion fun i => ↑(U i)) ((TopCat.GlueData.fromOpenSubset …
    -/
    rw [ι_fromOpenSubsetsGlue_apply]
    /-
      case h.mp.intro.intro.intro.mk
      α : Type u
      inst✝ : TopologicalSpace α
      J : Type u
      U : J → TopologicalSpace.Opens α
      i : (TopCat.GlueData.ofOpenSubsets U).J
      x : ↑(TopCat.of α)
      hx' : Membership.mem (U i) x
      ⊢ Membership.mem (Set.iUnion fun i => ↑(U i)) ((U i).inclusion' ⟨x, hx'⟩)
    -/
    exact Set.subset_iUnion _ i hx'
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      α : Type u
      inst✝ : TopologicalSpace α
      J : Type u
      U : J → TopologicalSpace.Opens α
      x✝ : ↑(TopCat.of α)
      ⊢ Membership.mem (Set.iUnion fun i => ↑(U i)) x✝ → Membership.mem (Set.range ⇑ …
    -/
  · rintro ⟨_, ⟨i, rfl⟩, hx⟩
    /-
      case h.mpr.intro.intro.intro
      α : Type u
      inst✝ : TopologicalSpace α
      J : Type u
      U : J → TopologicalSpace.Opens α
      x✝ : ↑(TopCat.of α)
      i : J
      hx : Membership.mem ((fun i => ↑(U i)) i) x✝
      ⊢ Membership.mem (Set.range ⇑(TopCat.GlueData.fromOpenSubsetsGlue U)) x✝
    -/
    rename_i x
    /-
      case h.mpr.intro.intro.intro
      α : Type u
      inst✝ : TopologicalSpace α
      J : Type u
      U : J → TopologicalSpace.Opens α
      x : ↑(TopCat.of α)
      i : J
      hx : Membership.mem ((fun i => ↑(U i)) i) x
      ⊢ Membership.mem (Set.range ⇑(TopCat.GlueData.fromOpenSubsetsGlue U)) x
    -/
    exact ⟨(ofOpenSubsets U).toGlueData.ι i ⟨x, hx⟩, ι_fromOpenSubsetsGlue_apply _ _ _⟩
    /-
      🎉 no goals
    -/


/-- The gluing of an open cover is homeomomorphic to the original space. -/
def openCoverGlueHomeo (h : ⋃ i, (U i : Set α) = Set.univ) :
    (ofOpenSubsets U).toGlueData.glued ≃ₜ α :=
  Homeomorph.homeomorphOfContinuousOpen
    (Equiv.ofBijective (fromOpenSubsetsGlue U)
      ⟨fromOpenSubsetsGlue_injective U,
        Set.range_eq_univ.mp ((range_fromOpenSubsetsGlue U).symm ▸ h)⟩)
    (fromOpenSubsetsGlue U).2 (fromOpenSubsetsGlue_isOpenMap U)


