/--
For a functor `G : C ⥤ D`, and a morphism `f : G.obj U ⟶ G.obj V`,
`Functor.imageSieve G f` is the sieve of `U`
consisting of those arrows whose composition with `f` has a lift in `G`.

This is the image sieve of `f` under `yonedaMap G V` and hence the name.
See `Functor.imageSieve_eq_imageSieve`.
-/
def Functor.imageSieve {U V : C} (f : G.obj U ⟶ G.obj V) : Sieve U where
  arrows _ i := ∃ l, G.map l = G.map i ≫ f
  downward_closed := by
    /-
      C : Type uC
      inst✝¹ : CategoryTheory.Category.{vC, uC} C
      D : Type uD
      inst✝ : CategoryTheory.Category.{vD, uD} D
      G : CategoryTheory.Functor C D
      J : CategoryTheory.GrothendieckTopology C
      K : CategoryTheory.GrothendieckTopology D
      U V : C
      f : Quiver.Hom (G.obj U) (G.obj V)
      ⊢ ∀ {Y Z : C} {f_1 : Quiver.Hom Y U}, (fun x i => Exists fun l => Eq (G.map l) …
    -/
    rintro Y₁ Y₂ i₁ ⟨l, hl⟩ i₂
    /-
      case intro
      C : Type uC
      inst✝¹ : CategoryTheory.Category.{vC, uC} C
      D : Type uD
      inst✝ : CategoryTheory.Category.{vD, uD} D
      G : CategoryTheory.Functor C D
      J : CategoryTheory.GrothendieckTopology C
      K : CategoryTheory.GrothendieckTopology D
      U V : C
      f : Quiver.Hom (G.obj U) (G.obj V)
      Y₁ Y₂ : C
      i₁ : Quiver.Hom Y₁ U
      l : Quiver.Hom Y₁ V
      hl : Eq (G.map l) (CategoryTheory.CategoryStruct.comp (G.map i₁) f)
      i₂ : Quiver.Hom Y₂ Y₁
      ⊢ Exists fun l => Eq (G.map l) (CategoryTheory.CategoryStruct.comp (G.map (Cat …
    -/
    exact ⟨i₂ ≫ l, by simp [hl]⟩
    /-
      🎉 no goals
    -/


@[simp]
lemma Functor.imageSieve_map {U V : C} (f : U ⟶ V) : G.imageSieve (G.map f) = ⊤ := by
  /-
    C : Type uC
    inst✝¹ : CategoryTheory.Category.{vC, uC} C
    D : Type uD
    inst✝ : CategoryTheory.Category.{vD, uD} D
    G : CategoryTheory.Functor C D
    U V : C
    f : Quiver.Hom U V
    ⊢ Eq (G.imageSieve (G.map f)) Top.top
  -/
  ext W g; simpa using ⟨g ≫ f, by simp⟩
           /-
             🎉 no goals
           -/


/--
For two arrows `f₁ f₂ : U ⟶ V`, the arrows `i` such that `i ≫ f₁ = i ≫ f₂` forms a sieve.
-/
@[simps]
def Sieve.equalizer {U V : C} (f₁ f₂ : U ⟶ V) : Sieve U where
  arrows _ i := i ≫ f₁ = i ≫ f₂
                        /-
                          C : Type uC
                          inst✝¹ : CategoryTheory.Category.{vC, uC} C
                          D : Type uD
                          inst✝ : CategoryTheory.Category.{vD, uD} D
                          G : CategoryTheory.Functor C D
                          J : CategoryTheory.GrothendieckTopology C
                          K : CategoryTheory.GrothendieckTopology D
                          U V : C
                          f₁ f₂ : Quiver.Hom U V
                          ⊢ ∀ {Y Z : C} {f : Quiver.Hom Y U}, (fun x i => Eq (CategoryTheory.CategoryStr …
                        -/
  downward_closed := by aesop
                        /-
                          🎉 no goals
                        -/


@[simp]
                                                                           /-
                                                                             C : Type uC
                                                                             inst✝ : CategoryTheory.Category.{vC, uC} C
                                                                             U V : C
                                                                             f : Quiver.Hom U V
                                                                             ⊢ Eq (CategoryTheory.Sieve.equalizer f f) Top.top
                                                                           -/
lemma Sieve.equalizer_self {U V : C} (f : U ⟶ V) : equalizer f f = ⊤ := by ext; simp
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/


lemma Sieve.equalizer_eq_equalizerSieve {U V : C} (f₁ f₂ : U ⟶ V) :
    Sieve.equalizer f₁ f₂ = Presheaf.equalizerSieve (F := yoneda.obj _) f₁ f₂ := rfl


lemma Functor.imageSieve_eq_imageSieve {D : Type uD} [Category.{vC} D] (G : C ⥤ D) {U V : C}
    (f : G.obj U ⟶ G.obj V) :
    G.imageSieve f = Presheaf.imageSieve (yonedaMap G V) f := rfl


/--
A functor `G : C ⥤ D` is locally full wrt a topology on `D` if for every `f : G.obj U ⟶ G.obj V`,
the set of `G.map fᵢ : G.obj Wᵢ ⟶ G.obj U` such that `G.map fᵢ ≫ f` is
in the image of `G` is a coverage of the topology on `D`.
-/
class IsLocallyFull : Prop where
  functorPushforward_imageSieve_mem : ∀ {U V} (f : G.obj U ⟶ G.obj V),
    (G.imageSieve f).functorPushforward G ∈ K _


/--
A functor `G : C ⥤ D` is locally faithful wrt a topology on `D` if for every `f₁ f₂ : U ⟶ V` whose
image in `D` are equal, the set of `G.map gᵢ : G.obj Wᵢ ⟶ G.obj U` such that `gᵢ ≫ f₁ = gᵢ ≫ f₂`
is a coverage of the topology on `D`.
-/
class IsLocallyFaithful : Prop where
  functorPushforward_equalizer_mem : ∀ {U V : C} (f₁ f₂ : U ⟶ V), G.map f₁ = G.map f₂ →
    (Sieve.equalizer f₁ f₂).functorPushforward G ∈ K _


lemma functorPushforward_imageSieve_mem [G.IsLocallyFull K] {U V} (f : G.obj U ⟶ G.obj V) :
    (G.imageSieve f).functorPushforward G ∈ K _ :=
  Functor.IsLocallyFull.functorPushforward_imageSieve_mem _


lemma functorPushforward_equalizer_mem
    [G.IsLocallyFaithful K] {U V} (f₁ f₂ : U ⟶ V) (e : G.map f₁ = G.map f₂) :
      (Sieve.equalizer f₁ f₂).functorPushforward G ∈ K _ :=
  Functor.IsLocallyFaithful.functorPushforward_equalizer_mem _ _ e


theorem IsLocallyFull.ext [G.IsLocallyFull K]
    (ℱ : Sheaf K (Type _)) {X Y : C} (i : G.obj X ⟶ G.obj Y)
    {s t : ℱ.val.obj (op (G.obj X))}
    (h : ∀ ⦃Z : C⦄ (j : Z ⟶ X) (f : Z ⟶ Y), G.map f = G.map j ≫ i →
      ℱ.1.map (G.map j).op s = ℱ.1.map (G.map j).op t) : s = t := by
  apply (((isSheaf_iff_isSheaf_of_type _ _).1 ℱ.cond) _
    (G.functorPushforward_imageSieve_mem K i)).isSeparatedFor.ext
  /-
    C : Type uC
    inst✝² : CategoryTheory.Category.{vC, uC} C
    D : Type uD
    inst✝¹ : CategoryTheory.Category.{vD, uD} D
    K : CategoryTheory.GrothendieckTopology D
    G : CategoryTheory.Functor C D
    inst✝ : G.IsLocallyFull K
    ℱ : CategoryTheory.Sheaf K (Type u_2)
    X Y : C
    i : Quiver.Hom (G.obj X) (G.obj Y)
    s t : ℱ.val.obj { unop := G.obj X }
    h : ∀ ⦃Z : C⦄ (j : Quiver.Hom Z X) (f : Quiver.Hom Z Y), Eq (G.map f) (Categor …
    ⊢ ∀ ⦃Y_1 : D⦄ ⦃f : Quiver.Hom Y_1 (G.obj X)⦄, (CategoryTheory.Sieve.functorPus …
  -/
  rintro Z _ ⟨W, iWX, iZW, ⟨iWY, e⟩, rfl⟩
  /-
    case intro.intro.intro.intro.intro
    C : Type uC
    inst✝² : CategoryTheory.Category.{vC, uC} C
    D : Type uD
    inst✝¹ : CategoryTheory.Category.{vD, uD} D
    K : CategoryTheory.GrothendieckTopology D
    G : CategoryTheory.Functor C D
    inst✝ : G.IsLocallyFull K
    ℱ : CategoryTheory.Sheaf K (Type u_2)
    X Y : C
    i : Quiver.Hom (G.obj X) (G.obj Y)
    s t : ℱ.val.obj { unop := G.obj X }
    h : ∀ ⦃Z : C⦄ (j : Quiver.Hom Z X) (f : Quiver.Hom Z Y), Eq (G.map f) (Categor …
    Z : D
    W : C
    iWX : Quiver.Hom W X
    iZW : Quiver.Hom Z (G.obj W)
    iWY : Quiver.Hom W Y
    e : Eq (G.map iWY) (CategoryTheory.CategoryStruct.comp (G.map iWX) i)
    ⊢ Eq (ℱ.val.map (CategoryTheory.CategoryStruct.comp iZW (G.map iWX)).op s) (ℱ. …
  -/
  simp [h iWX iWY e]
  /-
    🎉 no goals
  -/


theorem IsLocallyFaithful.ext [G.IsLocallyFaithful K] (ℱ : Sheaf K (Type _))
    {X Y : C} (i₁ i₂ : X ⟶ Y) (e : G.map i₁ = G.map i₂)
    {s t : ℱ.val.obj (op (G.obj X))}
    (h : ∀ ⦃Z : C⦄ (j : Z ⟶ X), j ≫ i₁ = j ≫ i₂ →
      ℱ.1.map (G.map j).op s = ℱ.1.map (G.map j).op t) : s = t := by
  apply (((isSheaf_iff_isSheaf_of_type _ _).1 ℱ.cond) _
    (G.functorPushforward_equalizer_mem K i₁ i₂ e)).isSeparatedFor.ext
  /-
    C : Type uC
    inst✝² : CategoryTheory.Category.{vC, uC} C
    D : Type uD
    inst✝¹ : CategoryTheory.Category.{vD, uD} D
    K : CategoryTheory.GrothendieckTopology D
    G : CategoryTheory.Functor C D
    inst✝ : G.IsLocallyFaithful K
    ℱ : CategoryTheory.Sheaf K (Type u_2)
    X Y : C
    i₁ i₂ : Quiver.Hom X Y
    e : Eq (G.map i₁) (G.map i₂)
    s t : ℱ.val.obj { unop := G.obj X }
    h : ∀ ⦃Z : C⦄ (j : Quiver.Hom Z X), Eq (CategoryTheory.CategoryStruct.comp j i …
    ⊢ ∀ ⦃Y_1 : D⦄ ⦃f : Quiver.Hom Y_1 (G.obj X)⦄, (CategoryTheory.Sieve.functorPus …
  -/
  rintro Z _ ⟨W, iWX, iZW, hiWX, rfl⟩
  /-
    case intro.intro.intro.intro
    C : Type uC
    inst✝² : CategoryTheory.Category.{vC, uC} C
    D : Type uD
    inst✝¹ : CategoryTheory.Category.{vD, uD} D
    K : CategoryTheory.GrothendieckTopology D
    G : CategoryTheory.Functor C D
    inst✝ : G.IsLocallyFaithful K
    ℱ : CategoryTheory.Sheaf K (Type u_2)
    X Y : C
    i₁ i₂ : Quiver.Hom X Y
    e : Eq (G.map i₁) (G.map i₂)
    s t : ℱ.val.obj { unop := G.obj X }
    h : ∀ ⦃Z : C⦄ (j : Quiver.Hom Z X), Eq (CategoryTheory.CategoryStruct.comp j i …
    Z : D
    W : C
    iWX : Quiver.Hom W X
    iZW : Quiver.Hom Z (G.obj W)
    hiWX : (CategoryTheory.Sieve.equalizer i₁ i₂).arrows iWX
    ⊢ Eq (ℱ.val.map (CategoryTheory.CategoryStruct.comp iZW (G.map iWX)).op s) (ℱ. …
  -/
  simp [h iWX hiWX]
  /-
    🎉 no goals
  -/


instance (priority := 900) IsLocallyFull.of_full [G.Full] : G.IsLocallyFull K where
  functorPushforward_imageSieve_mem f := by
    /-
      C : Type uC
      inst✝³ : CategoryTheory.Category.{vC, uC} C
      D : Type uD
      inst✝² : CategoryTheory.Category.{vD, uD} D
      G✝ : CategoryTheory.Functor C D
      J : CategoryTheory.GrothendieckTopology C
      K : CategoryTheory.GrothendieckTopology D
      A : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.11548, u_1} A
      G : CategoryTheory.Functor C D
      inst✝ : G.Full
      U✝ V✝ : C
      f : Quiver.Hom (G.obj U✝) (G.obj V✝)
      ⊢ Membership.mem (K (G.obj U✝)) (CategoryTheory.Sieve.functorPushforward G (G. …
    -/
    rw [← G.map_preimage f]
    /-
      C : Type uC
      inst✝³ : CategoryTheory.Category.{vC, uC} C
      D : Type uD
      inst✝² : CategoryTheory.Category.{vD, uD} D
      G✝ : CategoryTheory.Functor C D
      J : CategoryTheory.GrothendieckTopology C
      K : CategoryTheory.GrothendieckTopology D
      A : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.11548, u_1} A
      G : CategoryTheory.Functor C D
      inst✝ : G.Full
      U✝ V✝ : C
      f : Quiver.Hom (G.obj U✝) (G.obj V✝)
      ⊢ Membership.mem (K (G.obj U✝)) (CategoryTheory.Sieve.functorPushforward G (G. …
    -/
    simp only [Functor.imageSieve_map, Sieve.functorPushforward_top, GrothendieckTopology.top_mem]
    /-
      🎉 no goals
    -/


instance (priority := 900) IsLocallyFaithful.of_faithful [G.Faithful] : G.IsLocallyFaithful K where
                                                 /-
                                                   C : Type uC
                                                   inst✝³ : CategoryTheory.Category.{vC, uC} C
                                                   D : Type uD
                                                   inst✝² : CategoryTheory.Category.{vD, uD} D
                                                   G✝ : CategoryTheory.Functor C D
                                                   J : CategoryTheory.GrothendieckTopology C
                                                   K : CategoryTheory.GrothendieckTopology D
                                                   A : Type u_1
                                                   inst✝¹ : CategoryTheory.Category.{?u.12234, u_1} A
                                                   G : CategoryTheory.Functor C D
                                                   inst✝ : G.Faithful
                                                   U✝ V✝ : C
                                                   f₁ f₂ : Quiver.Hom U✝ V✝
                                                   e : Eq (G.map f₁) (G.map f₂)
                                                   ⊢ Membership.mem (K (G.obj U✝)) (CategoryTheory.Sieve.functorPushforward G (Ca …
                                                 -/
  functorPushforward_equalizer_mem f₁ f₂ e := by obtain rfl := G.map_injective e; simp
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/


