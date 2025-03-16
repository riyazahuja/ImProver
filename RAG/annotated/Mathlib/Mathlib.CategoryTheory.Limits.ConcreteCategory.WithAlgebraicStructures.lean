theorem colimit_rep_eq_zero
    (F : J ⥤ ModuleCat.{max t w} R) [PreservesColimit F (forget (ModuleCat R))] [IsFiltered J]
    [HasColimit F] (j : J) (x : F.obj j) (hx : colimit.ι F j x = 0) :
    ∃ (j' : J) (i : j ⟶ j'), (F.map i).hom x = 0 := by
  -- Break the abstraction barrier between homs and functions for `colimit_rep_eq_iff_exists`.
  have : ∀ (X Y : ModuleCat R) (f : X ⟶ Y),
    DFunLike.coe f.hom = DFunLike.coe (self := ConcreteCategory.instFunLike) f := fun _ _ _ => rfl
  /-
    R : Type u_1
    inst✝⁴ : Ring R
    J : Type w
    inst✝³ : CategoryTheory.Category.{r, w} J
    F : CategoryTheory.Functor J (ModuleCat R)
    inst✝² : CategoryTheory.Limits.PreservesColimit F (CategoryTheory.forget (Modu …
    inst✝¹ : CategoryTheory.IsFiltered J
    inst✝ : CategoryTheory.Limits.HasColimit F
    j : J
    x : ↑(F.obj j)
    hx : Eq ((CategoryTheory.Limits.colimit.ι F j).hom x) 0
    this : ∀ (X Y : ModuleCat R) (f : Quiver.Hom X Y), Eq ⇑f.hom ⇑f
    ⊢ Exists fun j' => Exists fun i => Eq ((F.map i).hom x) 0
  -/
  rw [show 0 = colimit.ι F j 0 by simp, this, colimit_rep_eq_iff_exists] at hx
  /-
    R : Type u_1
    inst✝⁴ : Ring R
    J : Type w
    inst✝³ : CategoryTheory.Category.{r, w} J
    F : CategoryTheory.Functor J (ModuleCat R)
    inst✝² : CategoryTheory.Limits.PreservesColimit F (CategoryTheory.forget (Modu …
    inst✝¹ : CategoryTheory.IsFiltered J
    inst✝ : CategoryTheory.Limits.HasColimit F
    j : J
    x : ↑(F.obj j)
    hx : Exists fun k => Exists fun f => Exists fun g => Eq ((F.map f) x) ((F.map  …
    this : ∀ (X Y : ModuleCat R) (f : Quiver.Hom X Y), Eq ⇑f.hom ⇑f
    ⊢ Exists fun j' => Exists fun i => Eq ((F.map i).hom x) 0
  -/
  obtain ⟨j', i, y, g⟩ := hx
  /-
    case intro.intro.intro
    R : Type u_1
    inst✝⁴ : Ring R
    J : Type w
    inst✝³ : CategoryTheory.Category.{r, w} J
    F : CategoryTheory.Functor J (ModuleCat R)
    inst✝² : CategoryTheory.Limits.PreservesColimit F (CategoryTheory.forget (Modu …
    inst✝¹ : CategoryTheory.IsFiltered J
    inst✝ : CategoryTheory.Limits.HasColimit F
    j : J
    x : ↑(F.obj j)
    this : ∀ (X Y : ModuleCat R) (f : Quiver.Hom X Y), Eq ⇑f.hom ⇑f
    j' : J
    i y : Quiver.Hom j j'
    g : Eq ((F.map i) x) ((F.map y) 0)
    ⊢ Exists fun j' => Exists fun i => Eq ((F.map i).hom x) 0
  -/
  exact ⟨j', i, g ▸ by simp [← this]⟩
  /-
    🎉 no goals
  -/


/--
if `r` has no zero smul divisors for all small-enough sections, then `r` has no zero smul divisors
in the colimit.
-/
lemma colimit_no_zero_smul_divisor
    (F : J ⥤ ModuleCat.{max t w} R) [PreservesColimit F (forget (ModuleCat R))]
    [IsFiltered J] [HasColimit F]
    (r : R) (H : ∃ (j' : J), ∀ (j : J) (_ : j' ⟶ j), ∀ (c : F.obj j), r • c = 0 → c = 0)
    (x : (forget (ModuleCat R)).obj (colimit F)) (hx : r • x = 0) : x = 0 := by

  -- Break the abstraction barrier between homs and functions for `Concrete.colimit_exists_rep`.
  have : ∀ (X Y : ModuleCat R) (f : X ⟶ Y),
    DFunLike.coe f.hom = DFunLike.coe (self := ConcreteCategory.instFunLike) f := fun _ _ _ => rfl
  classical
  obtain ⟨j, x, rfl⟩ := Concrete.colimit_exists_rep F x
  rw [← this, ← map_smul (colimit.ι F j).hom] at hx
  obtain ⟨j', i, h⟩ := Concrete.colimit_rep_eq_zero (hx := hx)
  obtain ⟨j'', H⟩ := H
  simpa [elementwise_of% (colimit.w F), this, map_zero] using congr(colimit.ι F _
    $(H (IsFiltered.sup {j, j', j''} { ⟨j, j', by simp, by simp, i⟩ })
      (IsFiltered.toSup _ _ <| by simp)
      (F.map (IsFiltered.toSup _ _ <| by simp) x)
      (by rw [← IsFiltered.toSup_commutes (f := i) (mY := by simp) (mf := by simp), F.map_comp,
        ModuleCat.comp_apply, ← map_smul, ← map_smul, h, map_zero])))


