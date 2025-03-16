/-- In a category `C` equipped with a shift by an additive monoid,
this is the type of morphisms `X ⟶ (Y⟦n⟧)` for `m : M`. -/
def ShiftedHom (X Y : C) (m : M) : Type _ := X ⟶ (Y⟦m⟧)


instance [Preadditive C] (X Y : C) (n : M) : AddCommGroup (ShiftedHom X Y n) := by
  /-
    C : Type u_1
    inst✝⁷ : CategoryTheory.Category.{?u.842, u_1} C
    D : Type u_2
    inst✝⁶ : CategoryTheory.Category.{?u.849, u_2} D
    E : Type u_3
    inst✝⁵ : CategoryTheory.Category.{?u.856, u_3} E
    M : Type u_4
    inst✝⁴ : AddMonoid M
    inst✝³ : CategoryTheory.HasShift C M
    inst✝² : CategoryTheory.HasShift D M
    inst✝¹ : CategoryTheory.HasShift E M
    inst✝ : CategoryTheory.Preadditive C
    X Y : C
    n : M
    ⊢ AddCommGroup (CategoryTheory.ShiftedHom X Y n)
  -/
  dsimp only [ShiftedHom]
  /-
    C : Type u_1
    inst✝⁷ : CategoryTheory.Category.{?u.842, u_1} C
    D : Type u_2
    inst✝⁶ : CategoryTheory.Category.{?u.849, u_2} D
    E : Type u_3
    inst✝⁵ : CategoryTheory.Category.{?u.856, u_3} E
    M : Type u_4
    inst✝⁴ : AddMonoid M
    inst✝³ : CategoryTheory.HasShift C M
    inst✝² : CategoryTheory.HasShift D M
    inst✝¹ : CategoryTheory.HasShift E M
    inst✝ : CategoryTheory.Preadditive C
    X Y : C
    n : M
    ⊢ AddCommGroup (Quiver.Hom X ((CategoryTheory.shiftFunctor C n).obj Y))
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- The composition of `f : X ⟶ Y⟦a⟧` and `g : Y ⟶ Z⟦b⟧`, as a morphism `X ⟶ Z⟦c⟧`
when `b + a = c`. -/
noncomputable def comp {a b c : M} (f : ShiftedHom X Y a) (g : ShiftedHom Y Z b) (h : b + a = c) :
    ShiftedHom X Z c :=
  f ≫ g⟦a⟧' ≫ (shiftFunctorAdd' C b a c h).inv.app _


lemma comp_assoc {a₁ a₂ a₃ a₁₂ a₂₃ a : M}
    (α : ShiftedHom X Y a₁) (β : ShiftedHom Y Z a₂) (γ : ShiftedHom Z T a₃)
    (h₁₂ : a₂ + a₁ = a₁₂) (h₂₃ : a₃ + a₂ = a₂₃) (h : a₃ + a₂ + a₁ = a) :
                                                /-
                                                  C : Type u_1
                                                  inst✝⁶ : CategoryTheory.Category.{?u.3238, u_1} C
                                                  D : Type u_2
                                                  inst✝⁵ : CategoryTheory.Category.{?u.3245, u_2} D
                                                  E : Type u_3
                                                  inst✝⁴ : CategoryTheory.Category.{?u.3252, u_3} E
                                                  M : Type u_4
                                                  inst✝³ : AddMonoid M
                                                  inst✝² : CategoryTheory.HasShift C M
                                                  inst✝¹ : CategoryTheory.HasShift D M
                                                  inst✝ : CategoryTheory.HasShift E M
                                                  X Y Z T : C
                                                  a₁ a₂ a₃ a₁₂ a₂₃ a : M
                                                  α : CategoryTheory.ShiftedHom X Y a₁
                                                  β : CategoryTheory.ShiftedHom Y Z a₂
                                                  γ : CategoryTheory.ShiftedHom Z T a₃
                                                  h₁₂ : Eq (HAdd.hAdd a₂ a₁) a₁₂
                                                  h₂₃ : Eq (HAdd.hAdd a₃ a₂) a₂₃
                                                  h : Eq (HAdd.hAdd (HAdd.hAdd a₃ a₂) a₁) a
                                                  ⊢ Eq (HAdd.hAdd a₃ a₁₂) a
                                                -/
    (α.comp β h₁₂).comp γ (show a₃ + a₁₂ = a by rw [← h₁₂, ← add_assoc, h]) =
                                                /-
                                                  🎉 no goals
                                                -/
                                /-
                                  C : Type u_1
                                  inst✝⁶ : CategoryTheory.Category.{?u.3238, u_1} C
                                  D : Type u_2
                                  inst✝⁵ : CategoryTheory.Category.{?u.3245, u_2} D
                                  E : Type u_3
                                  inst✝⁴ : CategoryTheory.Category.{?u.3252, u_3} E
                                  M : Type u_4
                                  inst✝³ : AddMonoid M
                                  inst✝² : CategoryTheory.HasShift C M
                                  inst✝¹ : CategoryTheory.HasShift D M
                                  inst✝ : CategoryTheory.HasShift E M
                                  X Y Z T : C
                                  a₁ a₂ a₃ a₁₂ a₂₃ a : M
                                  α : CategoryTheory.ShiftedHom X Y a₁
                                  β : CategoryTheory.ShiftedHom Y Z a₂
                                  γ : CategoryTheory.ShiftedHom Z T a₃
                                  h₁₂ : Eq (HAdd.hAdd a₂ a₁) a₁₂
                                  h₂₃ : Eq (HAdd.hAdd a₃ a₂) a₂₃
                                  h : Eq (HAdd.hAdd (HAdd.hAdd a₃ a₂) a₁) a
                                  ⊢ Eq (HAdd.hAdd a₂₃ a₁) a
                                -/
      α.comp (β.comp γ h₂₃) (by rw [← h₂₃, h]) := by
                                /-
                                  🎉 no goals
                                -/
  simp only [comp, assoc, Functor.map_comp,
    shiftFunctorAdd'_assoc_inv_app a₃ a₂ a₁ a₂₃ a₁₂ a h₂₃ h₁₂ h,
    ← NatTrans.naturality_assoc, Functor.comp_map]


/-- The element of `ShiftedHom X Y m₀` (when `m₀ = 0`) attached to a morphism `X ⟶ Y`. -/
noncomputable def mk₀ (m₀ : M) (hm₀ : m₀ = 0) (f : X ⟶ Y) : ShiftedHom X Y m₀ :=
  f ≫ (shiftFunctorZero' C m₀ hm₀).inv.app Y


/-- The bijection `(X ⟶ Y) ≃ ShiftedHom X Y m₀` when `m₀ = 0`. -/
@[simps apply]
noncomputable def homEquiv (m₀ : M) (hm₀ : m₀ = 0) : (X ⟶ Y) ≃ ShiftedHom X Y m₀ where
  toFun f := mk₀ m₀ hm₀ f
  invFun g := g ≫ (shiftFunctorZero' C m₀ hm₀).hom.app Y
                   /-
                     C : Type u_1
                     inst✝⁶ : CategoryTheory.Category.{?u.15747, u_1} C
                     D : Type u_2
                     inst✝⁵ : CategoryTheory.Category.{?u.15754, u_2} D
                     E : Type u_3
                     inst✝⁴ : CategoryTheory.Category.{?u.15761, u_3} E
                     M : Type u_4
                     inst✝³ : AddMonoid M
                     inst✝² : CategoryTheory.HasShift C M
                     inst✝¹ : CategoryTheory.HasShift D M
                     inst✝ : CategoryTheory.HasShift E M
                     X Y Z T : C
                     m₀ : M
                     hm₀ : Eq m₀ 0
                     f : Quiver.Hom X Y
                     ⊢ Eq ((fun g => CategoryTheory.CategoryStruct.comp g ((CategoryTheory.shiftFun …
                   -/
  left_inv f := by simp [mk₀]
                   /-
                     🎉 no goals
                   -/
                    /-
                      C : Type u_1
                      inst✝⁶ : CategoryTheory.Category.{?u.15747, u_1} C
                      D : Type u_2
                      inst✝⁵ : CategoryTheory.Category.{?u.15754, u_2} D
                      E : Type u_3
                      inst✝⁴ : CategoryTheory.Category.{?u.15761, u_3} E
                      M : Type u_4
                      inst✝³ : AddMonoid M
                      inst✝² : CategoryTheory.HasShift C M
                      inst✝¹ : CategoryTheory.HasShift D M
                      inst✝ : CategoryTheory.HasShift E M
                      X Y Z T : C
                      m₀ : M
                      hm₀ : Eq m₀ 0
                      g : CategoryTheory.ShiftedHom X Y m₀
                      ⊢ Eq ((fun f => CategoryTheory.ShiftedHom.mk₀ m₀ hm₀ f) ((fun g => CategoryThe …
                    -/
  right_inv g := by simp [mk₀]
                    /-
                      🎉 no goals
                    -/


lemma mk₀_comp (m₀ : M) (hm₀ : m₀ = 0) (f : X ⟶ Y) {a : M} (g : ShiftedHom Y Z a) :
                              /-
                                C : Type u_1
                                inst✝⁶ : CategoryTheory.Category.{?u.17328, u_1} C
                                D : Type u_2
                                inst✝⁵ : CategoryTheory.Category.{?u.17335, u_2} D
                                E : Type u_3
                                inst✝⁴ : CategoryTheory.Category.{?u.17342, u_3} E
                                M : Type u_4
                                inst✝³ : AddMonoid M
                                inst✝² : CategoryTheory.HasShift C M
                                inst✝¹ : CategoryTheory.HasShift D M
                                inst✝ : CategoryTheory.HasShift E M
                                X Y Z T : C
                                m₀ : M
                                hm₀ : Eq m₀ 0
                                f : Quiver.Hom X Y
                                a : M
                                g : CategoryTheory.ShiftedHom Y Z a
                                ⊢ Eq (HAdd.hAdd a m₀) a
                              -/
    (mk₀ m₀ hm₀ f).comp g (by rw [hm₀, add_zero]) = f ≫ g := by
                              /-
                                🎉 no goals
                              -/
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_5, u_1} C
    M : Type u_4
    inst✝¹ : AddMonoid M
    inst✝ : CategoryTheory.HasShift C M
    X Y Z : C
    m₀ : M
    hm₀ : Eq m₀ 0
    f : Quiver.Hom X Y
    a : M
    g : CategoryTheory.ShiftedHom Y Z a
    ⊢ Eq ((CategoryTheory.ShiftedHom.mk₀ m₀ hm₀ f).comp g ⋯) (CategoryTheory.Categ …
  -/
  subst hm₀
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_5, u_1} C
    M : Type u_4
    inst✝¹ : AddMonoid M
    inst✝ : CategoryTheory.HasShift C M
    X Y Z : C
    f : Quiver.Hom X Y
    a : M
    g : CategoryTheory.ShiftedHom Y Z a
    ⊢ Eq ((CategoryTheory.ShiftedHom.mk₀ 0 ⋯ f).comp g ⋯) (CategoryTheory.Category …
  -/
  simp [comp, mk₀, shiftFunctorAdd'_add_zero_inv_app, shiftFunctorZero']
  /-
    🎉 no goals
  -/


@[simp]
lemma mk₀_id_comp (m₀ : M) (hm₀ : m₀ = 0) {a : M} (f : ShiftedHom X Y a) :
                                  /-
                                    C : Type u_1
                                    inst✝⁶ : CategoryTheory.Category.{?u.19682, u_1} C
                                    D : Type u_2
                                    inst✝⁵ : CategoryTheory.Category.{?u.19689, u_2} D
                                    E : Type u_3
                                    inst✝⁴ : CategoryTheory.Category.{?u.19696, u_3} E
                                    M : Type u_4
                                    inst✝³ : AddMonoid M
                                    inst✝² : CategoryTheory.HasShift C M
                                    inst✝¹ : CategoryTheory.HasShift D M
                                    inst✝ : CategoryTheory.HasShift E M
                                    X Y Z T : C
                                    m₀ : M
                                    hm₀ : Eq m₀ 0
                                    a : M
                                    f : CategoryTheory.ShiftedHom X Y a
                                    ⊢ Eq (HAdd.hAdd a m₀) a
                                  -/
    (mk₀ m₀ hm₀ (𝟙 X)).comp f (by rw [hm₀, add_zero]) = f := by
                                  /-
                                    🎉 no goals
                                  -/
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_5, u_1} C
    M : Type u_4
    inst✝¹ : AddMonoid M
    inst✝ : CategoryTheory.HasShift C M
    X Y : C
    m₀ : M
    hm₀ : Eq m₀ 0
    a : M
    f : CategoryTheory.ShiftedHom X Y a
    ⊢ Eq ((CategoryTheory.ShiftedHom.mk₀ m₀ hm₀ (CategoryTheory.CategoryStruct.id  …
  -/
  simp [mk₀_comp]
  /-
    🎉 no goals
  -/


lemma comp_mk₀ {a : M} (f : ShiftedHom X Y a) (m₀ : M) (hm₀ : m₀ = 0) (g : Y ⟶ Z) :
                              /-
                                C : Type u_1
                                inst✝⁶ : CategoryTheory.Category.{?u.20916, u_1} C
                                D : Type u_2
                                inst✝⁵ : CategoryTheory.Category.{?u.20923, u_2} D
                                E : Type u_3
                                inst✝⁴ : CategoryTheory.Category.{?u.20930, u_3} E
                                M : Type u_4
                                inst✝³ : AddMonoid M
                                inst✝² : CategoryTheory.HasShift C M
                                inst✝¹ : CategoryTheory.HasShift D M
                                inst✝ : CategoryTheory.HasShift E M
                                X Y Z T : C
                                a : M
                                f : CategoryTheory.ShiftedHom X Y a
                                m₀ : M
                                hm₀ : Eq m₀ 0
                                g : Quiver.Hom Y Z
                                ⊢ Eq (HAdd.hAdd m₀ a) a
                              -/
    f.comp (mk₀ m₀ hm₀ g) (by rw [hm₀, zero_add]) = f ≫ g⟦a⟧' := by
                              /-
                                🎉 no goals
                              -/
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_5, u_1} C
    M : Type u_4
    inst✝¹ : AddMonoid M
    inst✝ : CategoryTheory.HasShift C M
    X Y Z : C
    a : M
    f : CategoryTheory.ShiftedHom X Y a
    m₀ : M
    hm₀ : Eq m₀ 0
    g : Quiver.Hom Y Z
    ⊢ Eq (f.comp (CategoryTheory.ShiftedHom.mk₀ m₀ hm₀ g) ⋯) (CategoryTheory.Categ …
  -/
  subst hm₀
  simp only [comp, shiftFunctorAdd'_zero_add_inv_app, mk₀, shiftFunctorZero',
    eqToIso_refl, Iso.refl_trans, ← Functor.map_comp, assoc, Iso.inv_hom_id_app,
    Functor.id_obj, comp_id]


@[simp]
lemma comp_mk₀_id {a : M} (f : ShiftedHom X Y a) (m₀ : M) (hm₀ : m₀ = 0) :
                                  /-
                                    C : Type u_1
                                    inst✝⁶ : CategoryTheory.Category.{?u.23892, u_1} C
                                    D : Type u_2
                                    inst✝⁵ : CategoryTheory.Category.{?u.23899, u_2} D
                                    E : Type u_3
                                    inst✝⁴ : CategoryTheory.Category.{?u.23906, u_3} E
                                    M : Type u_4
                                    inst✝³ : AddMonoid M
                                    inst✝² : CategoryTheory.HasShift C M
                                    inst✝¹ : CategoryTheory.HasShift D M
                                    inst✝ : CategoryTheory.HasShift E M
                                    X Y Z T : C
                                    a : M
                                    f : CategoryTheory.ShiftedHom X Y a
                                    m₀ : M
                                    hm₀ : Eq m₀ 0
                                    ⊢ Eq (HAdd.hAdd m₀ a) a
                                  -/
    f.comp (mk₀ m₀ hm₀ (𝟙 Y)) (by rw [hm₀, zero_add]) = f := by
                                  /-
                                    🎉 no goals
                                  -/
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_5, u_1} C
    M : Type u_4
    inst✝¹ : AddMonoid M
    inst✝ : CategoryTheory.HasShift C M
    X Y : C
    a : M
    f : CategoryTheory.ShiftedHom X Y a
    m₀ : M
    hm₀ : Eq m₀ 0
    ⊢ Eq (f.comp (CategoryTheory.ShiftedHom.mk₀ m₀ hm₀ (CategoryTheory.CategoryStr …
  -/
  simp [comp_mk₀]
  /-
    🎉 no goals
  -/


@[simp 1100]
lemma mk₀_comp_mk₀ (f : X ⟶ Y) (g : Y ⟶ Z) {a b c : M} (h : b + a = c)
    (ha : a = 0) (hb : b = 0) :
                                                 /-
                                                   C : Type u_1
                                                   inst✝⁶ : CategoryTheory.Category.{?u.25402, u_1} C
                                                   D : Type u_2
                                                   inst✝⁵ : CategoryTheory.Category.{?u.25409, u_2} D
                                                   E : Type u_3
                                                   inst✝⁴ : CategoryTheory.Category.{?u.25416, u_3} E
                                                   M : Type u_4
                                                   inst✝³ : AddMonoid M
                                                   inst✝² : CategoryTheory.HasShift C M
                                                   inst✝¹ : CategoryTheory.HasShift D M
                                                   inst✝ : CategoryTheory.HasShift E M
                                                   X Y Z T : C
                                                   f : Quiver.Hom X Y
                                                   g : Quiver.Hom Y Z
                                                   a b c : M
                                                   h : Eq (HAdd.hAdd b a) c
                                                   ha : Eq a 0
                                                   hb : Eq b 0
                                                   ⊢ Eq c 0
                                                 -/
    (mk₀ a ha f).comp (mk₀ b hb g) h = mk₀ c (by rw [← h, ha, hb, add_zero]) (f ≫ g) := by
                                                 /-
                                                   🎉 no goals
                                                 -/
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_5, u_1} C
    M : Type u_4
    inst✝¹ : AddMonoid M
    inst✝ : CategoryTheory.HasShift C M
    X Y Z : C
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    a b c : M
    h : Eq (HAdd.hAdd b a) c
    ha : Eq a 0
    hb : Eq b 0
    ⊢ Eq ((CategoryTheory.ShiftedHom.mk₀ a ha f).comp (CategoryTheory.ShiftedHom.m …
  -/
  subst ha hb
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_5, u_1} C
    M : Type u_4
    inst✝¹ : AddMonoid M
    inst✝ : CategoryTheory.HasShift C M
    X Y Z : C
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    c : M
    h : Eq (HAdd.hAdd 0 0) c
    ⊢ Eq ((CategoryTheory.ShiftedHom.mk₀ 0 ⋯ f).comp (CategoryTheory.ShiftedHom.mk …
  -/
  obtain rfl : c = 0 := by rw [← h, zero_add]
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_5, u_1} C
    M : Type u_4
    inst✝¹ : AddMonoid M
    inst✝ : CategoryTheory.HasShift C M
    X Y Z : C
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    h : Eq (HAdd.hAdd 0 0) 0
    ⊢ Eq ((CategoryTheory.ShiftedHom.mk₀ 0 ⋯ f).comp (CategoryTheory.ShiftedHom.mk …
  -/
  rw [mk₀_comp, mk₀, mk₀, assoc]
  /-
    🎉 no goals
  -/


@[simp]
lemma mk₀_comp_mk₀_assoc (f : X ⟶ Y) (g : Y ⟶ Z) {a : M}
    (ha : a = 0) {d : M} (h : ShiftedHom Z T d) :
    (mk₀ a ha f).comp ((mk₀ a ha g).comp h
                       /-
                         C : Type u_1
                         inst✝⁶ : CategoryTheory.Category.{?u.29263, u_1} C
                         D : Type u_2
                         inst✝⁵ : CategoryTheory.Category.{?u.29270, u_2} D
                         E : Type u_3
                         inst✝⁴ : CategoryTheory.Category.{?u.29277, u_3} E
                         M : Type u_4
                         inst✝³ : AddMonoid M
                         inst✝² : CategoryTheory.HasShift C M
                         inst✝¹ : CategoryTheory.HasShift D M
                         inst✝ : CategoryTheory.HasShift E M
                         X Y Z T : C
                         f : Quiver.Hom X Y
                         g : Quiver.Hom Y Z
                         a : M
                         ha : Eq a 0
                         d : M
                         h : CategoryTheory.ShiftedHom Z T d
                         ⊢ Eq (HAdd.hAdd d a) d
                       -/
                       /-
                         🎉 no goals
                       -/
        (show _ = d by rw [ha, add_zero])) (show _ = d by rw [ha, add_zero]) =
                                                          /-
                                                            🎉 no goals
                                                          -/
                                    /-
                                      C : Type u_1
                                      inst✝⁶ : CategoryTheory.Category.{?u.29263, u_1} C
                                      D : Type u_2
                                      inst✝⁵ : CategoryTheory.Category.{?u.29270, u_2} D
                                      E : Type u_3
                                      inst✝⁴ : CategoryTheory.Category.{?u.29277, u_3} E
                                      M : Type u_4
                                      inst✝³ : AddMonoid M
                                      inst✝² : CategoryTheory.HasShift C M
                                      inst✝¹ : CategoryTheory.HasShift D M
                                      inst✝ : CategoryTheory.HasShift E M
                                      X Y Z T : C
                                      f : Quiver.Hom X Y
                                      g : Quiver.Hom Y Z
                                      a : M
                                      ha : Eq a 0
                                      d : M
                                      h : CategoryTheory.ShiftedHom Z T d
                                      ⊢ Eq (HAdd.hAdd d a) d
                                    -/
      (mk₀ a ha (f ≫ g)).comp h (by rw [ha, add_zero]) := by
                                    /-
                                      🎉 no goals
                                    -/
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_5, u_1} C
    M : Type u_4
    inst✝¹ : AddMonoid M
    inst✝ : CategoryTheory.HasShift C M
    X Y Z T : C
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    a : M
    ha : Eq a 0
    d : M
    h : CategoryTheory.ShiftedHom Z T d
    ⊢ Eq ((CategoryTheory.ShiftedHom.mk₀ a ha f).comp ((CategoryTheory.ShiftedHom. …
  -/
  subst ha
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_5, u_1} C
    M : Type u_4
    inst✝¹ : AddMonoid M
    inst✝ : CategoryTheory.HasShift C M
    X Y Z T : C
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    d : M
    h : CategoryTheory.ShiftedHom Z T d
    ⊢ Eq ((CategoryTheory.ShiftedHom.mk₀ 0 ⋯ f).comp ((CategoryTheory.ShiftedHom.m …
  -/
  rw [← comp_assoc, mk₀_comp_mk₀]
  /-
    case h
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_5, u_1} C
    M : Type u_4
    inst✝¹ : AddMonoid M
    inst✝ : CategoryTheory.HasShift C M
    X Y Z T : C
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    d : M
    h : CategoryTheory.ShiftedHom Z T d
    ⊢ Eq (HAdd.hAdd 0 0) 0
  -/
  all_goals simp
  /-
    🎉 no goals
  -/


variable (X Y) in
@[simp]
                                                                          /-
                                                                            C : Type u_1
                                                                            inst✝³ : CategoryTheory.Category.{u_5, u_1} C
                                                                            M : Type u_4
                                                                            inst✝² : AddMonoid M
                                                                            inst✝¹ : CategoryTheory.HasShift C M
                                                                            X Y : C
                                                                            inst✝ : CategoryTheory.Preadditive C
                                                                            m₀ : M
                                                                            hm₀ : Eq m₀ 0
                                                                            ⊢ Eq (CategoryTheory.ShiftedHom.mk₀ m₀ hm₀ 0) 0
                                                                          -/
lemma mk₀_zero (m₀ : M) (hm₀ : m₀ = 0) : mk₀ m₀ hm₀ (0 : X ⟶ Y) = 0 := by simp [mk₀]
                                                                          /-
                                                                            🎉 no goals
                                                                          -/


@[simp]
lemma comp_add [∀ (a : M), (shiftFunctor C a).Additive]
    {a b c : M} (α : ShiftedHom X Y a) (β₁ β₂ : ShiftedHom Y Z b) (h : b + a = c) :
    α.comp (β₁ + β₂) h = α.comp β₁ h + α.comp β₂ h := by
  /-
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_5, u_1} C
    M : Type u_4
    inst✝³ : AddMonoid M
    inst✝² : CategoryTheory.HasShift C M
    X Y Z : C
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : ∀ (a : M), (CategoryTheory.shiftFunctor C a).Additive
    a b c : M
    α : CategoryTheory.ShiftedHom X Y a
    β₁ β₂ : CategoryTheory.ShiftedHom Y Z b
    h : Eq (HAdd.hAdd b a) c
    ⊢ Eq (α.comp (HAdd.hAdd β₁ β₂) h) (HAdd.hAdd (α.comp β₁ h) (α.comp β₂ h))
  -/
  rw [comp, comp, comp, Functor.map_add, Preadditive.add_comp, Preadditive.comp_add]
  /-
    🎉 no goals
  -/


@[simp]
lemma add_comp
    {a b c : M} (α₁ α₂ : ShiftedHom X Y a) (β : ShiftedHom Y Z b) (h : b + a = c) :
    (α₁ + α₂).comp β h = α₁.comp β h + α₂.comp β h := by
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_5, u_1} C
    M : Type u_4
    inst✝² : AddMonoid M
    inst✝¹ : CategoryTheory.HasShift C M
    X Y Z : C
    inst✝ : CategoryTheory.Preadditive C
    a b c : M
    α₁ α₂ : CategoryTheory.ShiftedHom X Y a
    β : CategoryTheory.ShiftedHom Y Z b
    h : Eq (HAdd.hAdd b a) c
    ⊢ Eq ((HAdd.hAdd α₁ α₂).comp β h) (HAdd.hAdd (α₁.comp β h) (α₂.comp β h))
  -/
  rw [comp, comp, comp, Preadditive.add_comp]
  /-
    🎉 no goals
  -/


@[simp]
lemma comp_neg [∀ (a : M), (shiftFunctor C a).Additive]
    {a b c : M} (α : ShiftedHom X Y a) (β : ShiftedHom Y Z b) (h : b + a = c) :
    α.comp (-β) h = -α.comp β h := by
  /-
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_5, u_1} C
    M : Type u_4
    inst✝³ : AddMonoid M
    inst✝² : CategoryTheory.HasShift C M
    X Y Z : C
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : ∀ (a : M), (CategoryTheory.shiftFunctor C a).Additive
    a b c : M
    α : CategoryTheory.ShiftedHom X Y a
    β : CategoryTheory.ShiftedHom Y Z b
    h : Eq (HAdd.hAdd b a) c
    ⊢ Eq (α.comp (Neg.neg β) h) (Neg.neg (α.comp β h))
  -/
  rw [comp, comp, Functor.map_neg, Preadditive.neg_comp, Preadditive.comp_neg]
  /-
    🎉 no goals
  -/


@[simp]
lemma neg_comp
    {a b c : M} (α : ShiftedHom X Y a) (β : ShiftedHom Y Z b) (h : b + a = c) :
    (-α).comp β h = -α.comp β h := by
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_5, u_1} C
    M : Type u_4
    inst✝² : AddMonoid M
    inst✝¹ : CategoryTheory.HasShift C M
    X Y Z : C
    inst✝ : CategoryTheory.Preadditive C
    a b c : M
    α : CategoryTheory.ShiftedHom X Y a
    β : CategoryTheory.ShiftedHom Y Z b
    h : Eq (HAdd.hAdd b a) c
    ⊢ Eq ((Neg.neg α).comp β h) (Neg.neg (α.comp β h))
  -/
  rw [comp, comp, Preadditive.neg_comp]
  /-
    🎉 no goals
  -/


variable (Z) in
@[simp]
lemma comp_zero [∀ (a : M), (shiftFunctor C a).PreservesZeroMorphisms]
    {a : M} (β : ShiftedHom X Y a) {b c : M} (h : b + a = c) :
    β.comp (0 : ShiftedHom Y Z b) h = 0 := by
  /-
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_5, u_1} C
    M : Type u_4
    inst✝³ : AddMonoid M
    inst✝² : CategoryTheory.HasShift C M
    X Y Z : C
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : ∀ (a : M), (CategoryTheory.shiftFunctor C a).PreservesZeroMorphisms
    a : M
    β : CategoryTheory.ShiftedHom X Y a
    b c : M
    h : Eq (HAdd.hAdd b a) c
    ⊢ Eq (β.comp 0 h) 0
  -/
  rw [comp, Functor.map_zero, Limits.zero_comp, Limits.comp_zero]
  /-
    🎉 no goals
  -/


variable (X) in
@[simp]
lemma zero_comp (a : M) {b c : M} (β : ShiftedHom Y Z b) (h : b + a = c) :
    (0 : ShiftedHom X Y a).comp β h = 0 := by
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_5, u_1} C
    M : Type u_4
    inst✝² : AddMonoid M
    inst✝¹ : CategoryTheory.HasShift C M
    X Y Z : C
    inst✝ : CategoryTheory.Preadditive C
    a b c : M
    β : CategoryTheory.ShiftedHom Y Z b
    h : Eq (HAdd.hAdd b a) c
    ⊢ Eq (CategoryTheory.ShiftedHom.comp 0 β h) 0
  -/
  rw [comp, Limits.zero_comp]
  /-
    🎉 no goals
  -/


/-- The action on `ShiftedHom` of a functor which commutes with the shift. -/
def map {a : M} (f : ShiftedHom X Y a) (F : C ⥤ D) [F.CommShift M] :
    ShiftedHom (F.obj X) (F.obj Y) a :=
  F.map f ≫ (F.commShiftIso a).hom.app Y


@[simp]
lemma id_map {a : M} (f : ShiftedHom X Y a) : f.map (𝟭 C) = f := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_5, u_1} C
    M : Type u_4
    inst✝¹ : AddMonoid M
    inst✝ : CategoryTheory.HasShift C M
    X Y : C
    a : M
    f : CategoryTheory.ShiftedHom X Y a
    ⊢ Eq (f.map (CategoryTheory.Functor.id C)) f
  -/
  simp [map, Functor.commShiftIso, Functor.CommShift.iso]
  /-
    🎉 no goals
  -/


lemma comp_map {a : M} (f : ShiftedHom X Y a) (F : C ⥤ D) [F.CommShift M]
    (G : D ⥤ E) [G.CommShift M] : f.map (F ⋙ G) = (f.map F).map G := by
  /-
    C : Type u_1
    inst✝⁸ : CategoryTheory.Category.{u_5, u_1} C
    D : Type u_2
    inst✝⁷ : CategoryTheory.Category.{u_6, u_2} D
    E : Type u_3
    inst✝⁶ : CategoryTheory.Category.{u_7, u_3} E
    M : Type u_4
    inst✝⁵ : AddMonoid M
    inst✝⁴ : CategoryTheory.HasShift C M
    inst✝³ : CategoryTheory.HasShift D M
    inst✝² : CategoryTheory.HasShift E M
    X Y : C
    a : M
    f : CategoryTheory.ShiftedHom X Y a
    F : CategoryTheory.Functor C D
    inst✝¹ : F.CommShift M
    G : CategoryTheory.Functor D E
    inst✝ : G.CommShift M
    ⊢ Eq (f.map (F.comp G)) ((f.map F).map G)
  -/
  simp [map, Functor.commShiftIso_comp_hom_app]
  /-
    🎉 no goals
  -/


lemma map_comp {a b c : M} (f : ShiftedHom X Y a) (g : ShiftedHom Y Z b)
    (h : b + a = c) (F : C ⥤ D) [F.CommShift M] :
    (f.comp g h).map F = (f.map F).comp (g.map F) h := by
  /-
    C : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_5, u_1} C
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_6, u_2} D
    M : Type u_4
    inst✝³ : AddMonoid M
    inst✝² : CategoryTheory.HasShift C M
    inst✝¹ : CategoryTheory.HasShift D M
    X Y Z : C
    a b c : M
    f : CategoryTheory.ShiftedHom X Y a
    g : CategoryTheory.ShiftedHom Y Z b
    h : Eq (HAdd.hAdd b a) c
    F : CategoryTheory.Functor C D
    inst✝ : F.CommShift M
    ⊢ Eq ((f.comp g h).map F) ((f.map F).comp (g.map F) h)
  -/
  dsimp [comp, map]
  /-
    C : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_5, u_1} C
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_6, u_2} D
    M : Type u_4
    inst✝³ : AddMonoid M
    inst✝² : CategoryTheory.HasShift C M
    inst✝¹ : CategoryTheory.HasShift D M
    X Y Z : C
    a b c : M
    f : CategoryTheory.ShiftedHom X Y a
    g : CategoryTheory.ShiftedHom Y Z b
    h : Eq (HAdd.hAdd b a) c
    F : CategoryTheory.Functor C D
    inst✝ : F.CommShift M
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map (CategoryTheory.CategoryStruct …
  -/
  simp only [Functor.map_comp, assoc]
  /-
    C : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_5, u_1} C
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_6, u_2} D
    M : Type u_4
    inst✝³ : AddMonoid M
    inst✝² : CategoryTheory.HasShift C M
    inst✝¹ : CategoryTheory.HasShift D M
    X Y Z : C
    a b c : M
    f : CategoryTheory.ShiftedHom X Y a
    g : CategoryTheory.ShiftedHom Y Z b
    h : Eq (HAdd.hAdd b a) c
    F : CategoryTheory.Functor C D
    inst✝ : F.CommShift M
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map f) (CategoryTheory.CategoryStr …
  -/
  erw [← NatTrans.naturality_assoc]
  simp only [Functor.comp_map, F.commShiftIso_add' h, Functor.CommShift.isoAdd'_hom_app,
    ← Functor.map_comp_assoc, Iso.inv_hom_id_app, Functor.comp_obj, comp_id, assoc]


