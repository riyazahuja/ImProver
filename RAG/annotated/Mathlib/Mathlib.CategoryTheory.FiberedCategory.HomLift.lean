/-- Helper-type for defining `IsHomLift`. -/
inductive IsHomLiftAux : ∀ {R S : 𝒮} {a b : 𝒳} (_ : R ⟶ S) (_ : a ⟶ b), Prop
  | map {a b : 𝒳} (φ : a ⟶ b) : IsHomLiftAux (p.map φ) φ


/-- Given a functor `p : 𝒳 ⥤ 𝒮`, an arrow `φ : a ⟶ b` in `𝒳` and an arrow `f : R ⟶ S` in `𝒮`,
`p.IsHomLift f φ` expresses the fact that `φ` lifts `f` through `p`.
This is often drawn as:
```
  a --φ--> b
  -        -
  |        |
  v        v
  R --f--> S
``` -/
class Functor.IsHomLift {R S : 𝒮} {a b : 𝒳} (f : R ⟶ S) (φ : a ⟶ b) : Prop where
  cond : IsHomLiftAux p f φ


/-- `subst_hom_lift p f φ` tries to substitute `f` with `p(φ)` by using `p.IsHomLift f φ` -/
macro "subst_hom_lift" p:term:max f:term:max φ:term:max : tactic =>
  `(tactic| obtain ⟨⟩ := Functor.IsHomLift.cond (p := $p) (f := $f) (φ := $φ))


/-- For any arrow `φ : a ⟶ b` in `𝒳`, `φ` lifts the arrow `p.map φ` in the base `𝒮`-/
@[simp]
instance {a b : 𝒳} (φ : a ⟶ b) : p.IsHomLift (p.map φ) φ where
             /-
               𝒮 : Type u₁
               𝒳 : Type u₂
               inst✝¹ : CategoryTheory.Category.{v₁, u₂} 𝒳
               inst✝ : CategoryTheory.Category.{v₂, u₁} 𝒮
               p : CategoryTheory.Functor 𝒳 𝒮
               a b : 𝒳
               φ : Quiver.Hom a b
               ⊢ CategoryTheory.IsHomLiftAux p (p.map φ) φ
             -/
  cond := by constructor
             /-
               🎉 no goals
             -/


@[simp]
instance (a : 𝒳) : p.IsHomLift (𝟙 (p.obj a)) (𝟙 a) := by
  /-
    𝒮 : Type u₁
    𝒳 : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₁, u₂} 𝒳
    inst✝ : CategoryTheory.Category.{v₂, u₁} 𝒮
    p : CategoryTheory.Functor 𝒳 𝒮
    a : 𝒳
    ⊢ p.IsHomLift (CategoryTheory.CategoryStruct.id (p.obj a)) (CategoryTheory.Cat …
  -/
  rw [← p.map_id]; infer_instance
                   /-
                     🎉 no goals
                   -/


protected lemma id {p : 𝒳 ⥤ 𝒮} {R : 𝒮} {a : 𝒳} (ha : p.obj a = R) : p.IsHomLift (𝟙 R) (𝟙 a) := by
  /-
    𝒮 : Type u₁
    𝒳 : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₁, u₂} 𝒳
    inst✝ : CategoryTheory.Category.{v₂, u₁} 𝒮
    p : CategoryTheory.Functor 𝒳 𝒮
    R : 𝒮
    a : 𝒳
    ha : Eq (p.obj a) R
    ⊢ p.IsHomLift (CategoryTheory.CategoryStruct.id R) (CategoryTheory.CategoryStr …
  -/
  cases ha; infer_instance
            /-
              🎉 no goals
            -/


lemma domain_eq (f : R ⟶ S) (φ : a ⟶ b) [p.IsHomLift f φ] : p.obj a = R := by
  /-
    𝒮 : Type u₁
    𝒳 : Type u₂
    inst✝² : CategoryTheory.Category.{v₁, u₂} 𝒳
    inst✝¹ : CategoryTheory.Category.{v₂, u₁} 𝒮
    p : CategoryTheory.Functor 𝒳 𝒮
    R S : 𝒮
    a b : 𝒳
    f : Quiver.Hom R S
    φ : Quiver.Hom a b
    inst✝ : p.IsHomLift f φ
    ⊢ Eq (p.obj a) R
  -/
  subst_hom_lift p f φ; rfl
                        /-
                          🎉 no goals
                        -/


lemma codomain_eq (f : R ⟶ S) (φ : a ⟶ b) [p.IsHomLift f φ] : p.obj b = S := by
  /-
    𝒮 : Type u₁
    𝒳 : Type u₂
    inst✝² : CategoryTheory.Category.{v₁, u₂} 𝒳
    inst✝¹ : CategoryTheory.Category.{v₂, u₁} 𝒮
    p : CategoryTheory.Functor 𝒳 𝒮
    R S : 𝒮
    a b : 𝒳
    f : Quiver.Hom R S
    φ : Quiver.Hom a b
    inst✝ : p.IsHomLift f φ
    ⊢ Eq (p.obj b) S
  -/
  subst_hom_lift p f φ; rfl
                        /-
                          🎉 no goals
                        -/


lemma fac : f = eqToHom (domain_eq p f φ).symm ≫ p.map φ ≫ eqToHom (codomain_eq p f φ) := by
  /-
    𝒮 : Type u₁
    𝒳 : Type u₂
    inst✝² : CategoryTheory.Category.{v₁, u₂} 𝒳
    inst✝¹ : CategoryTheory.Category.{v₂, u₁} 𝒮
    p : CategoryTheory.Functor 𝒳 𝒮
    R S : 𝒮
    a b : 𝒳
    f : Quiver.Hom R S
    φ : Quiver.Hom a b
    inst✝ : p.IsHomLift f φ
    ⊢ Eq f (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯) (Categor …
  -/
  subst_hom_lift p f φ; simp
                        /-
                          🎉 no goals
                        -/


lemma fac' : p.map φ = eqToHom (domain_eq p f φ) ≫ f ≫ eqToHom (codomain_eq p f φ).symm := by
  /-
    𝒮 : Type u₁
    𝒳 : Type u₂
    inst✝² : CategoryTheory.Category.{v₁, u₂} 𝒳
    inst✝¹ : CategoryTheory.Category.{v₂, u₁} 𝒮
    p : CategoryTheory.Functor 𝒳 𝒮
    R S : 𝒮
    a b : 𝒳
    f : Quiver.Hom R S
    φ : Quiver.Hom a b
    inst✝ : p.IsHomLift f φ
    ⊢ Eq (p.map φ) (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯)  …
  -/
  subst_hom_lift p f φ; simp
                        /-
                          🎉 no goals
                        -/


lemma commSq : CommSq (p.map φ) (eqToHom (domain_eq p f φ)) (eqToHom (codomain_eq p f φ)) f where
          /-
            𝒮 : Type u₁
            𝒳 : Type u₂
            inst✝² : CategoryTheory.Category.{v₁, u₂} 𝒳
            inst✝¹ : CategoryTheory.Category.{v₂, u₁} 𝒮
            p : CategoryTheory.Functor 𝒳 𝒮
            R S : 𝒮
            a b : 𝒳
            f : Quiver.Hom R S
            φ : Quiver.Hom a b
            inst✝ : p.IsHomLift f φ
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (p.map φ) (CategoryTheory.eqToHom ⋯)) …
          -/
  w := by simp only [fac p f φ, eqToHom_trans_assoc, eqToHom_refl, id_comp]
          /-
            🎉 no goals
          -/


lemma eq_of_isHomLift {a b : 𝒳} (f : p.obj a ⟶ p.obj b) (φ : a ⟶ b) [p.IsHomLift f φ] :
    f = p.map φ := by
  /-
    𝒮 : Type u₁
    𝒳 : Type u₂
    inst✝² : CategoryTheory.Category.{v₁, u₂} 𝒳
    inst✝¹ : CategoryTheory.Category.{v₂, u₁} 𝒮
    p : CategoryTheory.Functor 𝒳 𝒮
    a b : 𝒳
    f : Quiver.Hom (p.obj a) (p.obj b)
    φ : Quiver.Hom a b
    inst✝ : p.IsHomLift f φ
    ⊢ Eq f (p.map φ)
  -/
  simp only [fac p f φ, eqToHom_refl, comp_id, id_comp]
  /-
    🎉 no goals
  -/


lemma of_fac {R S : 𝒮} {a b : 𝒳} (f : R ⟶ S) (φ : a ⟶ b) (ha : p.obj a = R) (hb : p.obj b = S)
    (h : f = eqToHom ha.symm ≫ p.map φ ≫ eqToHom hb) : p.IsHomLift f φ := by
  /-
    𝒮 : Type u₁
    𝒳 : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₁, u₂} 𝒳
    inst✝ : CategoryTheory.Category.{v₂, u₁} 𝒮
    p : CategoryTheory.Functor 𝒳 𝒮
    R S : 𝒮
    a b : 𝒳
    f : Quiver.Hom R S
    φ : Quiver.Hom a b
    ha : Eq (p.obj a) R
    hb : Eq (p.obj b) S
    h : Eq f (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯) (Categ …
    ⊢ p.IsHomLift f φ
  -/
  subst ha hb h; simp
                 /-
                   🎉 no goals
                 -/


lemma of_fac' {R S : 𝒮} {a b : 𝒳} (f : R ⟶ S) (φ : a ⟶ b) (ha : p.obj a = R) (hb : p.obj b = S)
    (h : p.map φ = eqToHom ha ≫ f ≫ eqToHom hb.symm) : p.IsHomLift f φ := by
  /-
    𝒮 : Type u₁
    𝒳 : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₁, u₂} 𝒳
    inst✝ : CategoryTheory.Category.{v₂, u₁} 𝒮
    p : CategoryTheory.Functor 𝒳 𝒮
    R S : 𝒮
    a b : 𝒳
    f : Quiver.Hom R S
    φ : Quiver.Hom a b
    ha : Eq (p.obj a) R
    hb : Eq (p.obj b) S
    h : Eq (p.map φ) (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom h …
    ⊢ p.IsHomLift f φ
  -/
  subst ha hb
  /-
    𝒮 : Type u₁
    𝒳 : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₁, u₂} 𝒳
    inst✝ : CategoryTheory.Category.{v₂, u₁} 𝒮
    p : CategoryTheory.Functor 𝒳 𝒮
    a b : 𝒳
    φ : Quiver.Hom a b
    f : Quiver.Hom (p.obj a) (p.obj b)
    h : Eq (p.map φ) (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯ …
    ⊢ p.IsHomLift f φ
  -/
  obtain rfl : f = p.map φ := by simpa using h.symm
  /-
    𝒮 : Type u₁
    𝒳 : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₁, u₂} 𝒳
    inst✝ : CategoryTheory.Category.{v₂, u₁} 𝒮
    p : CategoryTheory.Functor 𝒳 𝒮
    a b : 𝒳
    φ : Quiver.Hom a b
    h : Eq (p.map φ) (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom ⋯ …
    ⊢ p.IsHomLift (p.map φ) φ
  -/
  infer_instance
  /-
    🎉 no goals
  -/


lemma of_commsq {R S : 𝒮} {a b : 𝒳} (f : R ⟶ S) (φ : a ⟶ b) (ha : p.obj a = R) (hb : p.obj b = S)
    (h : p.map φ ≫ eqToHom hb = (eqToHom ha) ≫ f) : p.IsHomLift f φ := by
  /-
    𝒮 : Type u₁
    𝒳 : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₁, u₂} 𝒳
    inst✝ : CategoryTheory.Category.{v₂, u₁} 𝒮
    p : CategoryTheory.Functor 𝒳 𝒮
    R S : 𝒮
    a b : 𝒳
    f : Quiver.Hom R S
    φ : Quiver.Hom a b
    ha : Eq (p.obj a) R
    hb : Eq (p.obj b) S
    h : Eq (CategoryTheory.CategoryStruct.comp (p.map φ) (CategoryTheory.eqToHom h …
    ⊢ p.IsHomLift f φ
  -/
  subst ha hb
  /-
    𝒮 : Type u₁
    𝒳 : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₁, u₂} 𝒳
    inst✝ : CategoryTheory.Category.{v₂, u₁} 𝒮
    p : CategoryTheory.Functor 𝒳 𝒮
    a b : 𝒳
    φ : Quiver.Hom a b
    f : Quiver.Hom (p.obj a) (p.obj b)
    h : Eq (CategoryTheory.CategoryStruct.comp (p.map φ) (CategoryTheory.eqToHom ⋯ …
    ⊢ p.IsHomLift f φ
  -/
  obtain rfl : f = p.map φ := by simpa using h.symm
  /-
    𝒮 : Type u₁
    𝒳 : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₁, u₂} 𝒳
    inst✝ : CategoryTheory.Category.{v₂, u₁} 𝒮
    p : CategoryTheory.Functor 𝒳 𝒮
    a b : 𝒳
    φ : Quiver.Hom a b
    h : Eq (CategoryTheory.CategoryStruct.comp (p.map φ) (CategoryTheory.eqToHom ⋯ …
    ⊢ p.IsHomLift (p.map φ) φ
  -/
  infer_instance
  /-
    🎉 no goals
  -/


lemma of_commSq {R S : 𝒮} {a b : 𝒳} (f : R ⟶ S) (φ : a ⟶ b) (ha : p.obj a = R) (hb : p.obj b = S)
    (h : CommSq (p.map φ) (eqToHom ha) (eqToHom hb) f) : p.IsHomLift f φ :=
  of_commsq p f φ ha hb h.1


instance comp {R S T : 𝒮} {a b c : 𝒳} (f : R ⟶ S) (g : S ⟶ T) (φ : a ⟶ b)
    (ψ : b ⟶ c) [p.IsHomLift f φ] [p.IsHomLift g ψ] : p.IsHomLift (f ≫ g) (φ ≫ ψ) := by
  /-
    𝒮 : Type u₁
    𝒳 : Type u₂
    inst✝³ : CategoryTheory.Category.{v₁, u₂} 𝒳
    inst✝² : CategoryTheory.Category.{v₂, u₁} 𝒮
    p : CategoryTheory.Functor 𝒳 𝒮
    R S T : 𝒮
    a b c : 𝒳
    f : Quiver.Hom R S
    g : Quiver.Hom S T
    φ : Quiver.Hom a b
    ψ : Quiver.Hom b c
    inst✝¹ : p.IsHomLift f φ
    inst✝ : p.IsHomLift g ψ
    ⊢ p.IsHomLift (CategoryTheory.CategoryStruct.comp f g) (CategoryTheory.Categor …
  -/
  apply of_commSq
  -- This line transforms the first goal in suitable form; the last line closes all three goals.
  /-
    case h
    𝒮 : Type u₁
    𝒳 : Type u₂
    inst✝³ : CategoryTheory.Category.{v₁, u₂} 𝒳
    inst✝² : CategoryTheory.Category.{v₂, u₁} 𝒮
    p : CategoryTheory.Functor 𝒳 𝒮
    R S T : 𝒮
    a b c : 𝒳
    f : Quiver.Hom R S
    g : Quiver.Hom S T
    φ : Quiver.Hom a b
    ψ : Quiver.Hom b c
    inst✝¹ : p.IsHomLift f φ
    inst✝ : p.IsHomLift g ψ
    ⊢ CategoryTheory.CommSq (p.map (CategoryTheory.CategoryStruct.comp φ ψ)) (Cate …
  -/
  on_goal 1 => rw [p.map_comp]
  /-
    case h
    𝒮 : Type u₁
    𝒳 : Type u₂
    inst✝³ : CategoryTheory.Category.{v₁, u₂} 𝒳
    inst✝² : CategoryTheory.Category.{v₂, u₁} 𝒮
    p : CategoryTheory.Functor 𝒳 𝒮
    R S T : 𝒮
    a b c : 𝒳
    f : Quiver.Hom R S
    g : Quiver.Hom S T
    φ : Quiver.Hom a b
    ψ : Quiver.Hom b c
    inst✝¹ : p.IsHomLift f φ
    inst✝ : p.IsHomLift g ψ
    ⊢ CategoryTheory.CommSq (CategoryTheory.CategoryStruct.comp (p.map φ) (p.map ψ …
  -/
  apply CommSq.horiz_comp (commSq p f φ) (commSq p g ψ)
  /-
    🎉 no goals
  -/


/-- If `φ : a ⟶ b` and `ψ : b ⟶ c` lift `𝟙 R`, then so does `φ ≫ ψ` -/
instance lift_id_comp (R : 𝒮) {a b c : 𝒳} (φ : a ⟶ b) (ψ : b ⟶ c)
    [p.IsHomLift (𝟙 R) φ] [p.IsHomLift (𝟙 R) ψ] : p.IsHomLift (𝟙 R) (φ ≫ ψ) :=
  comp_id (𝟙 R) ▸ comp p (𝟙 R) (𝟙 R) φ ψ


instance comp_lift_id_right {a b c : 𝒳} {S T : 𝒮} (f : S ⟶ T) (φ : a ⟶ b) [p.IsHomLift f φ]
    (ψ : b ⟶ c) [p.IsHomLift (𝟙 T) ψ] : p.IsHomLift f (φ ≫ ψ) := by
  /-
    𝒮 : Type u₁
    𝒳 : Type u₂
    inst✝³ : CategoryTheory.Category.{v₁, u₂} 𝒳
    inst✝² : CategoryTheory.Category.{v₂, u₁} 𝒮
    p : CategoryTheory.Functor 𝒳 𝒮
    a b c : 𝒳
    S T : 𝒮
    f : Quiver.Hom S T
    φ : Quiver.Hom a b
    inst✝¹ : p.IsHomLift f φ
    ψ : Quiver.Hom b c
    inst✝ : p.IsHomLift (CategoryTheory.CategoryStruct.id T) ψ
    ⊢ p.IsHomLift f (CategoryTheory.CategoryStruct.comp φ ψ)
  -/
  simpa using inferInstanceAs (p.IsHomLift (f ≫ 𝟙 T) (φ ≫ ψ))
  /-
    🎉 no goals
  -/


/-- If `φ : a ⟶ b` lifts `f` and `ψ : b ⟶ c` lifts `𝟙 T`, then `φ ≫ ψ` lifts `f` -/
lemma comp_lift_id_right' {R S : 𝒮} {a b c : 𝒳} (f : R ⟶ S) (φ : a ⟶ b) [p.IsHomLift f φ]
    (T : 𝒮) (ψ : b ⟶ c) [p.IsHomLift (𝟙 T) ψ] : p.IsHomLift f (φ ≫ ψ) := by
  /-
    𝒮 : Type u₁
    𝒳 : Type u₂
    inst✝³ : CategoryTheory.Category.{v₁, u₂} 𝒳
    inst✝² : CategoryTheory.Category.{v₂, u₁} 𝒮
    p : CategoryTheory.Functor 𝒳 𝒮
    R S : 𝒮
    a b c : 𝒳
    f : Quiver.Hom R S
    φ : Quiver.Hom a b
    inst✝¹ : p.IsHomLift f φ
    T : 𝒮
    ψ : Quiver.Hom b c
    inst✝ : p.IsHomLift (CategoryTheory.CategoryStruct.id T) ψ
    ⊢ p.IsHomLift f (CategoryTheory.CategoryStruct.comp φ ψ)
  -/
  obtain rfl : S = T := by rw [← codomain_eq p f φ, domain_eq p (𝟙 T) ψ]
  /-
    𝒮 : Type u₁
    𝒳 : Type u₂
    inst✝³ : CategoryTheory.Category.{v₁, u₂} 𝒳
    inst✝² : CategoryTheory.Category.{v₂, u₁} 𝒮
    p : CategoryTheory.Functor 𝒳 𝒮
    R S : 𝒮
    a b c : 𝒳
    f : Quiver.Hom R S
    φ : Quiver.Hom a b
    inst✝¹ : p.IsHomLift f φ
    ψ : Quiver.Hom b c
    inst✝ : p.IsHomLift (CategoryTheory.CategoryStruct.id S) ψ
    ⊢ p.IsHomLift f (CategoryTheory.CategoryStruct.comp φ ψ)
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance comp_lift_id_left {a b c : 𝒳} {S T : 𝒮} (f : S ⟶ T) (ψ : b ⟶ c) [p.IsHomLift f ψ]
    (φ : a ⟶ b) [p.IsHomLift (𝟙 S) φ] : p.IsHomLift f (φ ≫ ψ) := by
  /-
    𝒮 : Type u₁
    𝒳 : Type u₂
    inst✝³ : CategoryTheory.Category.{v₁, u₂} 𝒳
    inst✝² : CategoryTheory.Category.{v₂, u₁} 𝒮
    p : CategoryTheory.Functor 𝒳 𝒮
    a b c : 𝒳
    S T : 𝒮
    f : Quiver.Hom S T
    ψ : Quiver.Hom b c
    inst✝¹ : p.IsHomLift f ψ
    φ : Quiver.Hom a b
    inst✝ : p.IsHomLift (CategoryTheory.CategoryStruct.id S) φ
    ⊢ p.IsHomLift f (CategoryTheory.CategoryStruct.comp φ ψ)
  -/
  simpa using inferInstanceAs (p.IsHomLift (𝟙 S ≫ f) (φ ≫ ψ))
  /-
    🎉 no goals
  -/


/-- If `φ : a ⟶ b` lifts `𝟙 T` and `ψ : b ⟶ c` lifts `f`, then `φ  ≫ ψ` lifts `f` -/
lemma comp_lift_id_left' {a b c : 𝒳} (R : 𝒮) (φ : a ⟶ b) [p.IsHomLift (𝟙 R) φ]
    {S T : 𝒮} (f : S ⟶ T) (ψ : b ⟶ c) [p.IsHomLift f ψ] : p.IsHomLift f (φ ≫ ψ) := by
  /-
    𝒮 : Type u₁
    𝒳 : Type u₂
    inst✝³ : CategoryTheory.Category.{v₁, u₂} 𝒳
    inst✝² : CategoryTheory.Category.{v₂, u₁} 𝒮
    p : CategoryTheory.Functor 𝒳 𝒮
    a b c : 𝒳
    R : 𝒮
    φ : Quiver.Hom a b
    inst✝¹ : p.IsHomLift (CategoryTheory.CategoryStruct.id R) φ
    S T : 𝒮
    f : Quiver.Hom S T
    ψ : Quiver.Hom b c
    inst✝ : p.IsHomLift f ψ
    ⊢ p.IsHomLift f (CategoryTheory.CategoryStruct.comp φ ψ)
  -/
  obtain rfl : R = S := by rw [← codomain_eq p (𝟙 R) φ, domain_eq p f ψ]
  /-
    𝒮 : Type u₁
    𝒳 : Type u₂
    inst✝³ : CategoryTheory.Category.{v₁, u₂} 𝒳
    inst✝² : CategoryTheory.Category.{v₂, u₁} 𝒮
    p : CategoryTheory.Functor 𝒳 𝒮
    a b c : 𝒳
    R : 𝒮
    φ : Quiver.Hom a b
    inst✝¹ : p.IsHomLift (CategoryTheory.CategoryStruct.id R) φ
    T : 𝒮
    ψ : Quiver.Hom b c
    f : Quiver.Hom R T
    inst✝ : p.IsHomLift f ψ
    ⊢ p.IsHomLift f (CategoryTheory.CategoryStruct.comp φ ψ)
  -/
  infer_instance
  /-
    🎉 no goals
  -/


lemma eqToHom_domain_lift_id {p : 𝒳 ⥤ 𝒮} {a b : 𝒳} (hab : a = b) {R : 𝒮} (hR : p.obj a = R) :
    p.IsHomLift (𝟙 R) (eqToHom hab) := by
  /-
    𝒮 : Type u₁
    𝒳 : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₁, u₂} 𝒳
    inst✝ : CategoryTheory.Category.{v₂, u₁} 𝒮
    p : CategoryTheory.Functor 𝒳 𝒮
    a b : 𝒳
    hab : Eq a b
    R : 𝒮
    hR : Eq (p.obj a) R
    ⊢ p.IsHomLift (CategoryTheory.CategoryStruct.id R) (CategoryTheory.eqToHom hab)
  -/
  subst hR hab; simp
                /-
                  🎉 no goals
                -/


lemma eqToHom_codomain_lift_id {p : 𝒳 ⥤ 𝒮} {a b : 𝒳} (hab : a = b) {S : 𝒮} (hS : p.obj b = S) :
    p.IsHomLift (𝟙 S) (eqToHom hab) := by
  /-
    𝒮 : Type u₁
    𝒳 : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₁, u₂} 𝒳
    inst✝ : CategoryTheory.Category.{v₂, u₁} 𝒮
    p : CategoryTheory.Functor 𝒳 𝒮
    a b : 𝒳
    hab : Eq a b
    S : 𝒮
    hS : Eq (p.obj b) S
    ⊢ p.IsHomLift (CategoryTheory.CategoryStruct.id S) (CategoryTheory.eqToHom hab)
  -/
  subst hS hab; simp
                /-
                  🎉 no goals
                -/


lemma id_lift_eqToHom_domain {p : 𝒳 ⥤ 𝒮} {R S : 𝒮} (hRS : R = S) {a : 𝒳} (ha : p.obj a = R) :
    p.IsHomLift (eqToHom hRS) (𝟙 a) := by
  /-
    𝒮 : Type u₁
    𝒳 : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₁, u₂} 𝒳
    inst✝ : CategoryTheory.Category.{v₂, u₁} 𝒮
    p : CategoryTheory.Functor 𝒳 𝒮
    R S : 𝒮
    hRS : Eq R S
    a : 𝒳
    ha : Eq (p.obj a) R
    ⊢ p.IsHomLift (CategoryTheory.eqToHom hRS) (CategoryTheory.CategoryStruct.id a)
  -/
  subst hRS ha; simp
                /-
                  🎉 no goals
                -/


lemma id_lift_eqToHom_codomain {p : 𝒳 ⥤ 𝒮} {R S : 𝒮} (hRS : R = S) {b : 𝒳} (hb : p.obj b = S) :
    p.IsHomLift (eqToHom hRS) (𝟙 b) := by
  /-
    𝒮 : Type u₁
    𝒳 : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₁, u₂} 𝒳
    inst✝ : CategoryTheory.Category.{v₂, u₁} 𝒮
    p : CategoryTheory.Functor 𝒳 𝒮
    R S : 𝒮
    hRS : Eq R S
    b : 𝒳
    hb : Eq (p.obj b) S
    ⊢ p.IsHomLift (CategoryTheory.eqToHom hRS) (CategoryTheory.CategoryStruct.id b)
  -/
  subst hRS hb; simp
                /-
                  🎉 no goals
                -/


instance comp_eqToHom_lift {R S : 𝒮} {a' a b : 𝒳} (f : R ⟶ S) (φ : a ⟶ b) (h : a' = a)
    [p.IsHomLift f φ] : p.IsHomLift f (eqToHom h ≫ φ) := by
  /-
    𝒮 : Type u₁
    𝒳 : Type u₂
    inst✝² : CategoryTheory.Category.{v₁, u₂} 𝒳
    inst✝¹ : CategoryTheory.Category.{v₂, u₁} 𝒮
    p : CategoryTheory.Functor 𝒳 𝒮
    R S : 𝒮
    a' a b : 𝒳
    f : Quiver.Hom R S
    φ : Quiver.Hom a b
    h : Eq a' a
    inst✝ : p.IsHomLift f φ
    ⊢ p.IsHomLift f (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom h) …
  -/
  subst h; simp_all
           /-
             🎉 no goals
           -/


instance eqToHom_comp_lift {R S : 𝒮} {a b b' : 𝒳} (f : R ⟶ S) (φ : a ⟶ b) (h : b = b')
    [p.IsHomLift f φ] : p.IsHomLift f (φ ≫ eqToHom h) := by
  /-
    𝒮 : Type u₁
    𝒳 : Type u₂
    inst✝² : CategoryTheory.Category.{v₁, u₂} 𝒳
    inst✝¹ : CategoryTheory.Category.{v₂, u₁} 𝒮
    p : CategoryTheory.Functor 𝒳 𝒮
    R S : 𝒮
    a b b' : 𝒳
    f : Quiver.Hom R S
    φ : Quiver.Hom a b
    h : Eq b b'
    inst✝ : p.IsHomLift f φ
    ⊢ p.IsHomLift f (CategoryTheory.CategoryStruct.comp φ (CategoryTheory.eqToHom  …
  -/
  subst h; simp_all
           /-
             🎉 no goals
           -/


instance lift_eqToHom_comp {R' R S : 𝒮} {a b : 𝒳} (f : R ⟶ S) (φ : a ⟶ b) (h : R' = R)
    [p.IsHomLift f φ] : p.IsHomLift (eqToHom h ≫ f) φ := by
  /-
    𝒮 : Type u₁
    𝒳 : Type u₂
    inst✝² : CategoryTheory.Category.{v₁, u₂} 𝒳
    inst✝¹ : CategoryTheory.Category.{v₂, u₁} 𝒮
    p : CategoryTheory.Functor 𝒳 𝒮
    R' R S : 𝒮
    a b : 𝒳
    f : Quiver.Hom R S
    φ : Quiver.Hom a b
    h : Eq R' R
    inst✝ : p.IsHomLift f φ
    ⊢ p.IsHomLift (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom h) f …
  -/
  subst h; simp_all
           /-
             🎉 no goals
           -/


instance lift_comp_eqToHom {R S S' : 𝒮} {a b : 𝒳} (f : R ⟶ S) (φ : a ⟶ b) (h : S = S')
    [p.IsHomLift f φ] : p.IsHomLift (f ≫ eqToHom h) φ := by
  /-
    𝒮 : Type u₁
    𝒳 : Type u₂
    inst✝² : CategoryTheory.Category.{v₁, u₂} 𝒳
    inst✝¹ : CategoryTheory.Category.{v₂, u₁} 𝒮
    p : CategoryTheory.Functor 𝒳 𝒮
    R S S' : 𝒮
    a b : 𝒳
    f : Quiver.Hom R S
    φ : Quiver.Hom a b
    h : Eq S S'
    inst✝ : p.IsHomLift f φ
    ⊢ p.IsHomLift (CategoryTheory.CategoryStruct.comp f (CategoryTheory.eqToHom h) …
  -/
  subst h; simp_all
           /-
             🎉 no goals
           -/


@[simp]
lemma comp_eqToHom_lift_iff {R S : 𝒮} {a' a b : 𝒳} (f : R ⟶ S) (φ : a ⟶ b) (h : a' = a) :
    p.IsHomLift f (eqToHom h ≫ φ) ↔ p.IsHomLift f φ where
               /-
                 𝒮 : Type u₁
                 𝒳 : Type u₂
                 inst✝¹ : CategoryTheory.Category.{v₁, u₂} 𝒳
                 inst✝ : CategoryTheory.Category.{v₂, u₁} 𝒮
                 p : CategoryTheory.Functor 𝒳 𝒮
                 R S : 𝒮
                 a' a b : 𝒳
                 f : Quiver.Hom R S
                 φ : Quiver.Hom a b
                 h : Eq a' a
                 hφ' : p.IsHomLift f (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHo …
                 ⊢ p.IsHomLift f φ
               -/
  mp hφ' := by subst h; simpa using hφ'
                        /-
                          🎉 no goals
                        -/
  mpr _ := inferInstance


@[simp]
lemma eqToHom_comp_lift_iff {R S : 𝒮} {a b b' : 𝒳} (f : R ⟶ S) (φ : a ⟶ b) (h : b = b') :
    p.IsHomLift f (φ ≫ eqToHom h) ↔ p.IsHomLift f φ where
               /-
                 𝒮 : Type u₁
                 𝒳 : Type u₂
                 inst✝¹ : CategoryTheory.Category.{v₁, u₂} 𝒳
                 inst✝ : CategoryTheory.Category.{v₂, u₁} 𝒮
                 p : CategoryTheory.Functor 𝒳 𝒮
                 R S : 𝒮
                 a b b' : 𝒳
                 f : Quiver.Hom R S
                 φ : Quiver.Hom a b
                 h : Eq b b'
                 hφ' : p.IsHomLift f (CategoryTheory.CategoryStruct.comp φ (CategoryTheory.eqTo …
                 ⊢ p.IsHomLift f φ
               -/
  mp hφ' := by subst h; simpa using hφ'
                        /-
                          🎉 no goals
                        -/
  mpr _ := inferInstance


@[simp]
lemma lift_eqToHom_comp_iff {R' R S : 𝒮} {a b : 𝒳} (f : R ⟶ S) (φ : a ⟶ b) (h : R' = R) :
    p.IsHomLift (eqToHom h ≫ f) φ ↔ p.IsHomLift f φ where
               /-
                 𝒮 : Type u₁
                 𝒳 : Type u₂
                 inst✝¹ : CategoryTheory.Category.{v₁, u₂} 𝒳
                 inst✝ : CategoryTheory.Category.{v₂, u₁} 𝒮
                 p : CategoryTheory.Functor 𝒳 𝒮
                 R' R S : 𝒮
                 a b : 𝒳
                 f : Quiver.Hom R S
                 φ : Quiver.Hom a b
                 h : Eq R' R
                 hφ' : p.IsHomLift (CategoryTheory.CategoryStruct.comp (CategoryTheory.eqToHom  …
                 ⊢ p.IsHomLift f φ
               -/
  mp hφ' := by subst h; simpa using hφ'
                        /-
                          🎉 no goals
                        -/
  mpr _ := inferInstance


@[simp]
lemma lift_comp_eqToHom_iff {R S S' : 𝒮} {a b : 𝒳} (f : R ⟶ S) (φ : a ⟶ b) (h : S = S') :
    p.IsHomLift (f ≫ eqToHom h) φ ↔ p.IsHomLift f φ where
                      /-
                        𝒮 : Type u₁
                        𝒳 : Type u₂
                        inst✝¹ : CategoryTheory.Category.{v₁, u₂} 𝒳
                        inst✝ : CategoryTheory.Category.{v₂, u₁} 𝒮
                        p : CategoryTheory.Functor 𝒳 𝒮
                        R S S' : 𝒮
                        a b : 𝒳
                        f : Quiver.Hom R S
                        φ : Quiver.Hom a b
                        h : Eq S S'
                        hφ' : p.IsHomLift (CategoryTheory.CategoryStruct.comp f (CategoryTheory.eqToHo …
                        ⊢ p.IsHomLift f φ
                      -/
  mp := fun hφ' => by subst h; simpa using hφ'
                               /-
                                 🎉 no goals
                               -/
  mpr := fun _ => inferInstance


/-- Given a morphism `f : R ⟶ S`, and an isomorphism `φ : a ≅ b` lifting `f`, `isoOfIsoLift f φ` is
the isomorphism `Φ : R ≅ S` with `Φ.hom = f` induced from `φ` -/
@[simps hom]
def isoOfIsoLift (f : R ⟶ S) (φ : a ≅ b) [p.IsHomLift f φ.hom] :
    R ≅ S where
  hom := f
  inv := eqToHom (codomain_eq p f φ.hom).symm ≫ (p.mapIso φ).inv ≫ eqToHom (domain_eq p f φ.hom)
                   /-
                     𝒮 : Type u₁
                     𝒳 : Type u₂
                     inst✝² : CategoryTheory.Category.{v₁, u₂} 𝒳
                     inst✝¹ : CategoryTheory.Category.{v₂, u₁} 𝒮
                     p : CategoryTheory.Functor 𝒳 𝒮
                     R S : 𝒮
                     a b : 𝒳
                     f : Quiver.Hom R S
                     φ : CategoryTheory.Iso a b
                     inst✝ : p.IsHomLift f φ.hom
                     ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.CategoryStruct.comp …
                   -/
  hom_inv_id := by subst_hom_lift p f φ.hom; simp [← p.map_comp]
                                             /-
                                               🎉 no goals
                                             -/
                   /-
                     𝒮 : Type u₁
                     𝒳 : Type u₂
                     inst✝² : CategoryTheory.Category.{v₁, u₂} 𝒳
                     inst✝¹ : CategoryTheory.Category.{v₂, u₁} 𝒮
                     p : CategoryTheory.Functor 𝒳 𝒮
                     R S : 𝒮
                     a b : 𝒳
                     f : Quiver.Hom R S
                     φ : CategoryTheory.Iso a b
                     inst✝ : p.IsHomLift f φ.hom
                     ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
                   -/
  inv_hom_id := by subst_hom_lift p f φ.hom; simp [← p.map_comp]
                                             /-
                                               🎉 no goals
                                             -/


@[simp]
lemma isoOfIsoLift_inv_hom_id (f : R ⟶ S) (φ : a ≅ b) [p.IsHomLift f φ.hom] :
    (isoOfIsoLift p f φ).inv ≫ f = 𝟙 S :=
  (isoOfIsoLift p f φ).inv_hom_id


@[simp]
lemma isoOfIsoLift_hom_inv_id (f : R ⟶ S) (φ : a ≅ b) [p.IsHomLift f φ.hom] :
    f ≫ (isoOfIsoLift p f φ).inv = 𝟙 R :=
  (isoOfIsoLift p f φ).hom_inv_id


/-- If `φ : a ⟶ b` lifts `f : R ⟶ S` and `φ` is an isomorphism, then so is `f`. -/
lemma isIso_of_lift_isIso (f : R ⟶ S) (φ : a ⟶ b) [p.IsHomLift f φ] [IsIso φ] : IsIso f :=
  (fac p f φ) ▸ inferInstance


/-- Given `φ : a ≅ b` and `f : R ≅ S`, such that `φ.hom` lifts `f.hom`, then `φ.inv` lifts
`f.inv`. -/
instance inv_lift_inv (f : R ≅ S) (φ : a ≅ b) [p.IsHomLift f.hom φ.hom] :
    p.IsHomLift f.inv φ.inv := by
  /-
    𝒮 : Type u₁
    𝒳 : Type u₂
    inst✝² : CategoryTheory.Category.{v₁, u₂} 𝒳
    inst✝¹ : CategoryTheory.Category.{v₂, u₁} 𝒮
    p : CategoryTheory.Functor 𝒳 𝒮
    R S : 𝒮
    a b : 𝒳
    f : CategoryTheory.Iso R S
    φ : CategoryTheory.Iso a b
    inst✝ : p.IsHomLift f.hom φ.hom
    ⊢ p.IsHomLift f.inv φ.inv
  -/
  apply of_commSq
  /-
    case h
    𝒮 : Type u₁
    𝒳 : Type u₂
    inst✝² : CategoryTheory.Category.{v₁, u₂} 𝒳
    inst✝¹ : CategoryTheory.Category.{v₂, u₁} 𝒮
    p : CategoryTheory.Functor 𝒳 𝒮
    R S : 𝒮
    a b : 𝒳
    f : CategoryTheory.Iso R S
    φ : CategoryTheory.Iso a b
    inst✝ : p.IsHomLift f.hom φ.hom
    ⊢ CategoryTheory.CommSq (p.map φ.inv) (CategoryTheory.eqToHom ?ha) (CategoryTh …
  -/
  apply CommSq.horiz_inv (f := p.mapIso φ) (commSq p f.hom φ.hom)
  /-
    🎉 no goals
  -/


/-- Given `φ : a ≅ b` and `f : R ⟶ S`, such that `φ.hom` lifts `f`, then `φ.inv` lifts the
inverse of `f` given by `isoOfIsoLift`. -/
instance inv_lift (f : R ⟶ S) (φ : a ≅ b) [p.IsHomLift f φ.hom] :
    p.IsHomLift (isoOfIsoLift p f φ).inv φ.inv := by
  /-
    𝒮 : Type u₁
    𝒳 : Type u₂
    inst✝² : CategoryTheory.Category.{v₁, u₂} 𝒳
    inst✝¹ : CategoryTheory.Category.{v₂, u₁} 𝒮
    p : CategoryTheory.Functor 𝒳 𝒮
    R S : 𝒮
    a b : 𝒳
    f : Quiver.Hom R S
    φ : CategoryTheory.Iso a b
    inst✝ : p.IsHomLift f φ.hom
    ⊢ p.IsHomLift (CategoryTheory.IsHomLift.isoOfIsoLift p f φ).inv φ.inv
  -/
  apply of_commSq
  /-
    case h
    𝒮 : Type u₁
    𝒳 : Type u₂
    inst✝² : CategoryTheory.Category.{v₁, u₂} 𝒳
    inst✝¹ : CategoryTheory.Category.{v₂, u₁} 𝒮
    p : CategoryTheory.Functor 𝒳 𝒮
    R S : 𝒮
    a b : 𝒳
    f : Quiver.Hom R S
    φ : CategoryTheory.Iso a b
    inst✝ : p.IsHomLift f φ.hom
    ⊢ CategoryTheory.CommSq (p.map φ.inv) (CategoryTheory.eqToHom ?ha) (CategoryTh …
  -/
  apply CommSq.horiz_inv (f := p.mapIso φ) (by apply commSq p f φ.hom)
  /-
    🎉 no goals
  -/


/-- If `φ : a ⟶ b` lifts `f : R ⟶ S` and both are isomorphisms, then `φ⁻¹` lifts `f⁻¹`. -/
protected instance inv (f : R ⟶ S) (φ : a ⟶ b) [IsIso f] [IsIso φ] [p.IsHomLift f φ] :
    p.IsHomLift (inv f) (inv φ) :=
                                                       /-
                                                         𝒮 : Type u₁
                                                         𝒳 : Type u₂
                                                         inst✝⁴ : CategoryTheory.Category.{v₁, u₂} 𝒳
                                                         inst✝³ : CategoryTheory.Category.{v₂, u₁} 𝒮
                                                         p : CategoryTheory.Functor 𝒳 𝒮
                                                         R S : 𝒮
                                                         a b : 𝒳
                                                         f : Quiver.Hom R S
                                                         φ : Quiver.Hom a b
                                                         inst✝² : CategoryTheory.IsIso f
                                                         inst✝¹ : CategoryTheory.IsIso φ
                                                         inst✝ : p.IsHomLift f φ
                                                         ⊢ p.IsHomLift (CategoryTheory.asIso f).hom (CategoryTheory.asIso φ).hom
                                                       -/
  have : p.IsHomLift (asIso f).hom (asIso φ).hom := by simp_all
                                                       /-
                                                         🎉 no goals
                                                       -/
  IsHomLift.inv_lift_inv p (asIso f) (asIso φ)


/-- If `φ : a ≅ b` is an isomorphism lifting `𝟙 S` for some `S : 𝒮`, then `φ⁻¹` also
lifts `𝟙 S`. -/
instance lift_id_inv (S : 𝒮) {a b : 𝒳} (φ : a ≅ b) [p.IsHomLift (𝟙 S) φ.hom] :
    p.IsHomLift (𝟙 S) φ.inv :=
                                                   /-
                                                     𝒮 : Type u₁
                                                     𝒳 : Type u₂
                                                     inst✝² : CategoryTheory.Category.{v₁, u₂} 𝒳
                                                     inst✝¹ : CategoryTheory.Category.{v₂, u₁} 𝒮
                                                     p : CategoryTheory.Functor 𝒳 𝒮
                                                     S : 𝒮
                                                     a b : 𝒳
                                                     φ : CategoryTheory.Iso a b
                                                     inst✝ : p.IsHomLift (CategoryTheory.CategoryStruct.id S) φ.hom
                                                     ⊢ p.IsHomLift (CategoryTheory.asIso (CategoryTheory.CategoryStruct.id S)).hom  …
                                                   -/
  have : p.IsHomLift (asIso (𝟙 S)).hom φ.hom := by simp_all
                                                   /-
                                                     🎉 no goals
                                                   -/
  (IsIso.inv_id (X := S)) ▸ (IsHomLift.inv_lift_inv p (asIso (𝟙 S)) φ)


instance lift_id_inv_isIso (S : 𝒮) {a b : 𝒳} (φ : a ⟶ b) [IsIso φ] [p.IsHomLift (𝟙 S) φ] :
    p.IsHomLift (𝟙 S) (inv φ) :=
  (IsIso.inv_id (X := S)) ▸ (IsHomLift.inv p _ φ)


