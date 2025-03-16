@[reassoc (attr := simp)]
lemma idem_f (P : Karoubi (Karoubi C)) : P.p.f ≫ P.p.f = P.p.f := by
  /-
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    P : CategoryTheory.Idempotents.Karoubi (CategoryTheory.Idempotents.Karoubi C)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp P.p.f P.p.f) P.p.f
  -/
  simpa only [hom_ext_iff, comp_f] using P.idem
  /-
    🎉 no goals
  -/


@[reassoc]
lemma p_comm_f {P Q : Karoubi (Karoubi C)} (f : P ⟶ Q) : P.p.f ≫ f.f.f = f.f.f ≫ Q.p.f := by
  /-
    C : Type u_1
    inst✝ : CategoryTheory.Category.{u_2, u_1} C
    P Q : CategoryTheory.Idempotents.Karoubi (CategoryTheory.Idempotents.Karoubi C)
    f : Quiver.Hom P Q
    ⊢ Eq (CategoryTheory.CategoryStruct.comp P.p.f f.f.f) (CategoryTheory.Category …
  -/
  simpa only [hom_ext_iff, comp_f] using p_comm f
  /-
    🎉 no goals
  -/


/-- The canonical functor `Karoubi (Karoubi C) ⥤ Karoubi C` -/
@[simps]
def inverse : Karoubi (Karoubi C) ⥤ Karoubi C where
                             /-
                               C : Type u_1
                               inst✝ : CategoryTheory.Category.{?u.1706, u_1} C
                               P : CategoryTheory.Idempotents.Karoubi (CategoryTheory.Idempotents.Karoubi C)
                               ⊢ Eq (CategoryTheory.CategoryStruct.comp P.p.f P.p.f) P.p.f
                             -/
  obj P := ⟨P.X.X, P.p.f, by simpa only [hom_ext_iff] using P.idem⟩
                             /-
                               🎉 no goals
                             -/
                      /-
                        C : Type u_1
                        inst✝ : CategoryTheory.Category.{?u.1706, u_1} C
                        X✝ Y✝ : CategoryTheory.Idempotents.Karoubi (CategoryTheory.Idempotents.Karoubi …
                        f : Quiver.Hom X✝ Y✝
                        ⊢ Eq f.f.f (CategoryTheory.CategoryStruct.comp ((fun P => { X := P.X.X, p := P …
                      -/
  map f := ⟨f.f.f, by simpa only [hom_ext_iff] using f.comm⟩
                      /-
                        🎉 no goals
                      -/


instance [Preadditive C] : Functor.Additive (inverse C) where


/-- The unit isomorphism of the equivalence -/
@[simps!]
def unitIso : 𝟭 (Karoubi C) ≅ toKaroubi (Karoubi C) ⋙ inverse C :=
                           /-
                             C : Type u_1
                             inst✝ : CategoryTheory.Category.{?u.5977, u_1} C
                             ⊢ ∀ (X : CategoryTheory.Idempotents.Karoubi C), Eq ((CategoryTheory.Functor.id …
                           -/
                           /-
                             🎉 no goals
                           -/
  eqToIso (Functor.ext (by aesop_cat) (by aesop_cat))
                                          /-
                                            🎉 no goals
                                          -/


attribute [local simp] p_comm_f in
/-- The counit isomorphism of the equivalence -/
@[simps]
def counitIso : inverse C ⋙ toKaroubi (Karoubi C) ≅ 𝟭 (Karoubi (Karoubi C)) where
  hom := { app := fun P => { f := { f := P.p.1 } } }
  inv := { app := fun P => { f := { f := P.p.1 }  } }


/-- The equivalence `Karoubi C ≌ Karoubi (Karoubi C)` -/
@[simps]
def equivalence : Karoubi C ≌ Karoubi (Karoubi C) where
  functor := toKaroubi (Karoubi C)
  inverse := KaroubiKaroubi.inverse C
  unitIso := KaroubiKaroubi.unitIso C
  counitIso := KaroubiKaroubi.counitIso C


instance equivalence.additive_functor [Preadditive C] :
  Functor.Additive (equivalence C).functor where


instance equivalence.additive_inverse [Preadditive C] :
  Functor.Additive (equivalence C).inverse where


