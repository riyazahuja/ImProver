/--
`X.AffineZariskiSite` is the small affine Zariski site of `X`, whose elements are affine open
sets of `X`, and whose arrows are basic open sets `D(f) ⟶ U` for any `f : Γ(X, U)`.

Note that this differs from the definition on stacks project where the arrows in the small affine
Zariski site are arbitrary inclusions.
-/
def Scheme.AffineZariskiSite (X : Scheme.{u}) : Type u := { U : X.Opens // IsAffineOpen U }


/-- The inclusion from `X.AffineZariskiSite` to `X.Opens`. -/
abbrev toOpens (U : X.AffineZariskiSite) : X.Opens := U.1


instance : Preorder X.AffineZariskiSite where
  le U V := ∃ f : Γ(X, V.toOpens), X.basicOpen f = U.toOpens
  le_refl U := ⟨1, Scheme.basicOpen_of_isUnit _ isUnit_one⟩
  le_trans := by
    /-
      X : AlgebraicGeometry.Scheme
      ⊢ ∀ (a b c : X.AffineZariskiSite), LE.le a b → LE.le b c → LE.le a c
    -/
    rintro ⟨U, hU⟩ ⟨V, hV⟩ ⟨W, hW⟩ ⟨f, rfl⟩ ⟨g, rfl⟩
    /-
      case mk.mk.mk.intro.intro
      X : AlgebraicGeometry.Scheme
      W : X.Opens
      hW : AlgebraicGeometry.IsAffineOpen W
      g : ↑(X.presheaf.obj { unop := AlgebraicGeometry.Scheme.AffineZariskiSite.toOp …
      hV : AlgebraicGeometry.IsAffineOpen (X.basicOpen g)
      f : ↑(X.presheaf.obj { unop := AlgebraicGeometry.Scheme.AffineZariskiSite.toOp …
      hU : AlgebraicGeometry.IsAffineOpen (X.basicOpen f)
      ⊢ LE.le ⟨X.basicOpen f, hU⟩ ⟨W, hW⟩
    -/
    exact hW.basicOpen_basicOpen_is_basicOpen g f
    /-
      🎉 no goals
    -/


lemma toOpens_mono :
    Monotone (toOpens (X := X)) := by
  /-
    X : AlgebraicGeometry.Scheme
    ⊢ Monotone AlgebraicGeometry.Scheme.AffineZariskiSite.toOpens
  -/
  rintro ⟨U, hU⟩ ⟨V, hV⟩ ⟨f, rfl⟩
  /-
    case mk.mk.intro
    X : AlgebraicGeometry.Scheme
    V : X.Opens
    hV : AlgebraicGeometry.IsAffineOpen V
    f : ↑(X.presheaf.obj { unop := AlgebraicGeometry.Scheme.AffineZariskiSite.toOp …
    hU : AlgebraicGeometry.IsAffineOpen (X.basicOpen f)
    ⊢ LE.le (AlgebraicGeometry.Scheme.AffineZariskiSite.toOpens ⟨X.basicOpen f, hU …
  -/
  exact X.basicOpen_le _
  /-
    🎉 no goals
  -/


lemma toOpens_injective : Function.Injective (toOpens (X := X)) := Subtype.val_injective


instance : PartialOrder X.AffineZariskiSite where
  le_antisymm _ _ hUV hVU := Subtype.ext ((toOpens_mono hUV).antisymm (toOpens_mono hVU))


/-- The basic open set of a section, as an element of `AffineZariskiSite`. -/
def basicOpen (U : X.AffineZariskiSite) (f : Γ(X, U.toOpens)) : X.AffineZariskiSite :=
  ⟨X.basicOpen f, U.2.basicOpen f⟩


lemma basicOpen_le (U : X.AffineZariskiSite) (f : Γ(X, U.toOpens)) : U.basicOpen f ≤ U :=
  ⟨f, rfl⟩


variable (X) in
/-- The inclusion functor from `X.AffineZariskiSite` to `X.Opens`. -/
def toOpensFunctor : X.AffineZariskiSite ⥤ X.Opens := toOpens_mono.functor


instance : (toOpensFunctor X).Faithful where


instance : (toOpensFunctor X).IsLocallyFull (Opens.grothendieckTopology X) where
  functorPushforward_imageSieve_mem := by
    /-
      X : AlgebraicGeometry.Scheme
      ⊢ ∀ {U V : X.AffineZariskiSite} (f : Quiver.Hom ((AlgebraicGeometry.Scheme.Aff …
    -/
    intro U V h x hx
    /-
      X : AlgebraicGeometry.Scheme
      U V : X.AffineZariskiSite
      h : Quiver.Hom ((AlgebraicGeometry.Scheme.AffineZariskiSite.toOpensFunctor X). …
      x : ↑↑X.toPresheafedSpace
      hx : Membership.mem ((AlgebraicGeometry.Scheme.AffineZariskiSite.toOpensFuncto …
      ⊢ Exists fun U_1 => Exists fun f => And ((CategoryTheory.Sieve.functorPushforw …
    -/
    obtain ⟨f, hfU, hxf⟩ := V.2.exists_basicOpen_le ⟨x, hx⟩ (h.le hx)
    exact ⟨X.basicOpen f, homOfLE hfU, ⟨V.basicOpen f,
      ⟨_, (X.basicOpen_res f h.op).trans (inf_eq_right.mpr hfU)⟩, 𝟙 _,
      ⟨⟨f, rfl⟩, rfl⟩, rfl⟩, hxf⟩


instance : (toOpensFunctor X).IsCoverDense (Opens.grothendieckTopology X) where
  is_cover := by
    /-
      X : AlgebraicGeometry.Scheme
      ⊢ ∀ (U : X.Opens), Membership.mem ((Opens.grothendieckTopology ↑↑X.toPresheafe …
    -/
    intros U x hx
    /-
      X : AlgebraicGeometry.Scheme
      U : X.Opens
      x : ↑↑X.toPresheafedSpace
      hx : Membership.mem U x
      ⊢ Exists fun U_1 => Exists fun f => And ((CategoryTheory.Sieve.coverByImage (A …
    -/
    obtain ⟨_, ⟨V, hV, rfl⟩, hxV, hVU⟩ := (isBasis_affine_open X).exists_subset_of_mem_open hx U.2
    /-
      case intro.intro.intro.intro.intro
      X : AlgebraicGeometry.Scheme
      U : X.Opens
      x : ↑↑X.toPresheafedSpace
      hx : Membership.mem U x
      V : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
      hV : Membership.mem X.affineOpens V
      hxV : Membership.mem (↑V) x
      hVU : HasSubset.Subset ↑V ↑U
      ⊢ Exists fun U_1 => Exists fun f => And ((CategoryTheory.Sieve.coverByImage (A …
    -/
    exact ⟨V, homOfLE hVU, ⟨⟨V, hV⟩, 𝟙 _, homOfLE hVU, rfl⟩, hxV⟩
    /-
      🎉 no goals
    -/


variable (X) in
/-- The grothendieck topology on `X.AffineZariskiSite` induced from the topology on `X.Opens`.
Also see `mem_grothendieckTopology_iff_sectionsOfPresieve`. -/
def grothendieckTopology : GrothendieckTopology X.AffineZariskiSite :=
  (toOpensFunctor X).inducedTopology (Opens.grothendieckTopology X)


lemma mem_grothendieckTopology {U : X.AffineZariskiSite} {S : Sieve U} :
    S ∈ grothendieckTopology X U ↔
      ∀ x ∈ U.toOpens, ∃ (V : _) (f : V ⟶ U), S.arrows f ∧ x ∈ V.toOpens := by
  /-
    X : AlgebraicGeometry.Scheme
    U : X.AffineZariskiSite
    S : CategoryTheory.Sieve U
    ⊢ Iff (Membership.mem ((AlgebraicGeometry.Scheme.AffineZariskiSite.grothendiec …
  -/
  apply forall₂_congr fun x hxU ↦ ⟨?_, ?_⟩
    /-
      X : AlgebraicGeometry.Scheme
      U : X.AffineZariskiSite
      S : CategoryTheory.Sieve U
      x : ↑↑X.toPresheafedSpace
      hxU : Membership.mem ((AlgebraicGeometry.Scheme.AffineZariskiSite.toOpensFunct …
      ⊢ (Exists fun U_1 => Exists fun f => And ((CategoryTheory.Sieve.functorPushfor …
    -/
  · rintro ⟨V, f, ⟨W, g, h, hg, rfl⟩, hxV⟩
    /-
      case intro.intro.intro.intro.intro.intro.intro
      X : AlgebraicGeometry.Scheme
      U : X.AffineZariskiSite
      S : CategoryTheory.Sieve U
      x : ↑↑X.toPresheafedSpace
      hxU : Membership.mem ((AlgebraicGeometry.Scheme.AffineZariskiSite.toOpensFunct …
      V : TopologicalSpace.Opens ↑↑X.toPresheafedSpace
      hxV : Membership.mem V x
      W : X.AffineZariskiSite
      g : Quiver.Hom W U
      h : Quiver.Hom V ((AlgebraicGeometry.Scheme.AffineZariskiSite.toOpensFunctor X …
      hg : S.arrows g
      ⊢ Exists fun V => Exists fun f => And (S.arrows f) (Membership.mem V.toOpens x)
    -/
    exact ⟨W, g, hg, h.le hxV⟩
    /-
      🎉 no goals
    -/
    /-
      X : AlgebraicGeometry.Scheme
      U : X.AffineZariskiSite
      S : CategoryTheory.Sieve U
      x : ↑↑X.toPresheafedSpace
      hxU : Membership.mem ((AlgebraicGeometry.Scheme.AffineZariskiSite.toOpensFunct …
      ⊢ (Exists fun V => Exists fun f => And (S.arrows f) (Membership.mem V.toOpens  …
    -/
  · rintro ⟨W, g, hg, hxW⟩
    /-
      case intro.intro.intro
      X : AlgebraicGeometry.Scheme
      U : X.AffineZariskiSite
      S : CategoryTheory.Sieve U
      x : ↑↑X.toPresheafedSpace
      hxU : Membership.mem ((AlgebraicGeometry.Scheme.AffineZariskiSite.toOpensFunct …
      W : X.AffineZariskiSite
      g : Quiver.Hom W U
      hg : S.arrows g
      hxW : Membership.mem W.toOpens x
      ⊢ Exists fun U_1 => Exists fun f => And ((CategoryTheory.Sieve.functorPushforw …
    -/
    exact ⟨W.toOpens, homOfLE (toOpens_mono g.le), ⟨W, g, 𝟙 _, hg, rfl⟩, hxW⟩
    /-
      🎉 no goals
    -/


instance : (toOpensFunctor X).IsDenseSubsite
    (grothendieckTopology X) (Opens.grothendieckTopology X) where
  functorPushforward_mem_iff := Iff.rfl


/-- The presieve associated to a set of sections.
This is a surjection, see `presieveOfSections_surjective`. -/
def presieveOfSections (U : X.AffineZariskiSite) (s : Set Γ(X, U.toOpens)) : Presieve U :=
  fun V _ ↦ ∃ f ∈ s, X.basicOpen f = V.toOpens


/-- The set of sections associated to a presieve. -/
def sectionsOfPresieve {U : X.AffineZariskiSite} (P : Presieve U) : Set Γ(X, U.toOpens) :=
  { f | P (homOfLE (U.basicOpen_le f)) }


lemma presieveOfSections_sectionsOfPresieve {U : X.AffineZariskiSite} (P : Presieve U) :
    presieveOfSections U (sectionsOfPresieve P) = P := by
  /-
    X : AlgebraicGeometry.Scheme
    U : X.AffineZariskiSite
    P : CategoryTheory.Presieve U
    ⊢ Eq (U.presieveOfSections (AlgebraicGeometry.Scheme.AffineZariskiSite.section …
  -/
  refine funext₂ fun ⟨V, hV⟩ ⟨f, hf⟩ ↦ eq_iff_iff.mpr ⟨?_, ?_⟩
    /-
      case refine_1
      X : AlgebraicGeometry.Scheme
      U : X.AffineZariskiSite
      P : CategoryTheory.Presieve U
      x✝¹ : X.AffineZariskiSite
      V : X.Opens
      hV : AlgebraicGeometry.IsAffineOpen V
      x✝ : Quiver.Hom ⟨V, hV⟩ U
      f : ↑(X.presheaf.obj { unop := U.toOpens })
      hf : Eq (X.basicOpen f) (AlgebraicGeometry.Scheme.AffineZariskiSite.toOpens ⟨V …
      ⊢ U.presieveOfSections (AlgebraicGeometry.Scheme.AffineZariskiSite.sectionsOfP …
    -/
  · rintro ⟨_, H, rfl⟩
    /-
      case refine_1.intro.intro
      X : AlgebraicGeometry.Scheme
      U : X.AffineZariskiSite
      P : CategoryTheory.Presieve U
      x✝¹ : X.AffineZariskiSite
      f w✝ : ↑(X.presheaf.obj { unop := U.toOpens })
      H : Membership.mem (AlgebraicGeometry.Scheme.AffineZariskiSite.sectionsOfPresi …
      hV : AlgebraicGeometry.IsAffineOpen (X.basicOpen w✝)
      x✝ : Quiver.Hom ⟨X.basicOpen w✝, hV⟩ U
      hf : Eq (X.basicOpen f) (AlgebraicGeometry.Scheme.AffineZariskiSite.toOpens ⟨X …
      ⊢ P { down := { down := ⋯ } }
    -/
    exact H
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      X : AlgebraicGeometry.Scheme
      U : X.AffineZariskiSite
      P : CategoryTheory.Presieve U
      x✝¹ : X.AffineZariskiSite
      V : X.Opens
      hV : AlgebraicGeometry.IsAffineOpen V
      x✝ : Quiver.Hom ⟨V, hV⟩ U
      f : ↑(X.presheaf.obj { unop := U.toOpens })
      hf : Eq (X.basicOpen f) (AlgebraicGeometry.Scheme.AffineZariskiSite.toOpens ⟨V …
      ⊢ P { down := { down := ⋯ } } → U.presieveOfSections (AlgebraicGeometry.Scheme …
    -/
  · intro H
    /-
      case refine_2
      X : AlgebraicGeometry.Scheme
      U : X.AffineZariskiSite
      P : CategoryTheory.Presieve U
      x✝¹ : X.AffineZariskiSite
      V : X.Opens
      hV : AlgebraicGeometry.IsAffineOpen V
      x✝ : Quiver.Hom ⟨V, hV⟩ U
      f : ↑(X.presheaf.obj { unop := U.toOpens })
      hf : Eq (X.basicOpen f) (AlgebraicGeometry.Scheme.AffineZariskiSite.toOpens ⟨V …
      H : P { down := { down := ⋯ } }
      ⊢ U.presieveOfSections (AlgebraicGeometry.Scheme.AffineZariskiSite.sectionsOfP …
    -/
    obtain rfl : _ = V := hf
    /-
      case refine_2
      X : AlgebraicGeometry.Scheme
      U : X.AffineZariskiSite
      P : CategoryTheory.Presieve U
      x✝¹ : X.AffineZariskiSite
      f : ↑(X.presheaf.obj { unop := U.toOpens })
      hV : AlgebraicGeometry.IsAffineOpen (X.basicOpen f)
      x✝ : Quiver.Hom ⟨X.basicOpen f, hV⟩ U
      H : P { down := { down := ⋯ } }
      ⊢ U.presieveOfSections (AlgebraicGeometry.Scheme.AffineZariskiSite.sectionsOfP …
    -/
    exact ⟨_, H, rfl⟩
    /-
      🎉 no goals
    -/


lemma presieveOfSections_surjective {U : X.AffineZariskiSite} :
    Function.Surjective (presieveOfSections U) :=
  fun _ ↦ ⟨_, presieveOfSections_sectionsOfPresieve _⟩


lemma presieveOfSections_eq_ofArrows (U : X.AffineZariskiSite) (s : Set Γ(X, U.toOpens)) :
    presieveOfSections U s = .ofArrows _ (fun i : s ↦ homOfLE (U.basicOpen_le i.1)) := by
  /-
    X : AlgebraicGeometry.Scheme
    U : X.AffineZariskiSite
    s : Set ↑(X.presheaf.obj { unop := U.toOpens })
    ⊢ Eq (U.presieveOfSections s) (CategoryTheory.Presieve.ofArrows (fun i => U.ba …
  -/
  refine funext₂ fun ⟨V, hV⟩ ⟨f, hf⟩ ↦ eq_iff_iff.mpr ⟨?_, ?_⟩
    /-
      case refine_1
      X : AlgebraicGeometry.Scheme
      U : X.AffineZariskiSite
      s : Set ↑(X.presheaf.obj { unop := U.toOpens })
      x✝¹ : X.AffineZariskiSite
      V : X.Opens
      hV : AlgebraicGeometry.IsAffineOpen V
      x✝ : Quiver.Hom ⟨V, hV⟩ U
      f : ↑(X.presheaf.obj { unop := U.toOpens })
      hf : Eq (X.basicOpen f) (AlgebraicGeometry.Scheme.AffineZariskiSite.toOpens ⟨V …
      ⊢ U.presieveOfSections s { down := { down := ⋯ } } → CategoryTheory.Presieve.o …
    -/
  · rintro ⟨f, hfs, rfl⟩
    /-
      case refine_1.intro.intro
      X : AlgebraicGeometry.Scheme
      U : X.AffineZariskiSite
      s : Set ↑(X.presheaf.obj { unop := U.toOpens })
      x✝¹ : X.AffineZariskiSite
      f✝ f : ↑(X.presheaf.obj { unop := U.toOpens })
      hfs : Membership.mem s f
      hV : AlgebraicGeometry.IsAffineOpen (X.basicOpen f)
      x✝ : Quiver.Hom ⟨X.basicOpen f, hV⟩ U
      hf : Eq (X.basicOpen f✝) (AlgebraicGeometry.Scheme.AffineZariskiSite.toOpens ⟨ …
      ⊢ CategoryTheory.Presieve.ofArrows (fun i => U.basicOpen ↑i) (fun i => Categor …
    -/
    exact .mk (ι := s) ⟨f, hfs⟩
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      X : AlgebraicGeometry.Scheme
      U : X.AffineZariskiSite
      s : Set ↑(X.presheaf.obj { unop := U.toOpens })
      x✝¹ : X.AffineZariskiSite
      V : X.Opens
      hV : AlgebraicGeometry.IsAffineOpen V
      x✝ : Quiver.Hom ⟨V, hV⟩ U
      f : ↑(X.presheaf.obj { unop := U.toOpens })
      hf : Eq (X.basicOpen f) (AlgebraicGeometry.Scheme.AffineZariskiSite.toOpens ⟨V …
      ⊢ CategoryTheory.Presieve.ofArrows (fun i => U.basicOpen ↑i) (fun i => Categor …
    -/
  · rintro ⟨⟨f, hfs⟩⟩
    /-
      case refine_2.mk.mk
      X : AlgebraicGeometry.Scheme
      U : X.AffineZariskiSite
      s : Set ↑(X.presheaf.obj { unop := U.toOpens })
      x✝¹ : X.AffineZariskiSite
      f✝ : ↑(X.presheaf.obj { unop := U.toOpens })
      Y : X.AffineZariskiSite
      f : ↑(X.presheaf.obj { unop := U.toOpens })
      hfs : Membership.mem s f
      hV : AlgebraicGeometry.IsAffineOpen (X.basicOpen ↑⟨f, hfs⟩)
      x✝ : Quiver.Hom ⟨X.basicOpen ↑⟨f, hfs⟩, hV⟩ U
      hf : Eq (X.basicOpen f✝) (AlgebraicGeometry.Scheme.AffineZariskiSite.toOpens ⟨ …
      ⊢ U.presieveOfSections s { down := { down := ⋯ } }
    -/
    exact ⟨f, hfs, rfl⟩
    /-
      🎉 no goals
    -/


lemma generate_presieveOfSections
    {U V : X.AffineZariskiSite} {s : Set Γ(X, U.toOpens)} {f : V ⟶ U} :
    Sieve.generate (presieveOfSections U s) f ↔ ∃ f ∈ s, ∃ g, X.basicOpen (f * g) = V.toOpens := by
  /-
    X : AlgebraicGeometry.Scheme
    U V : X.AffineZariskiSite
    s : Set ↑(X.presheaf.obj { unop := U.toOpens })
    f : Quiver.Hom V U
    ⊢ Iff ((CategoryTheory.Sieve.generate (U.presieveOfSections s)).arrows f) (Exi …
  -/
  obtain ⟨V, hV⟩ := V
  /-
    case mk
    X : AlgebraicGeometry.Scheme
    U : X.AffineZariskiSite
    s : Set ↑(X.presheaf.obj { unop := U.toOpens })
    V : X.Opens
    hV : AlgebraicGeometry.IsAffineOpen V
    f : Quiver.Hom ⟨V, hV⟩ U
    ⊢ Iff ((CategoryTheory.Sieve.generate (U.presieveOfSections s)).arrows f) (Exi …
  -/
  constructor
    /-
      case mk.mp
      X : AlgebraicGeometry.Scheme
      U : X.AffineZariskiSite
      s : Set ↑(X.presheaf.obj { unop := U.toOpens })
      V : X.Opens
      hV : AlgebraicGeometry.IsAffineOpen V
      f : Quiver.Hom ⟨V, hV⟩ U
      ⊢ (CategoryTheory.Sieve.generate (U.presieveOfSections s)).arrows f → Exists f …
    -/
  · rintro ⟨⟨W, hW⟩, ⟨f₁, hf₁⟩, -, ⟨f₂, hf₂s, rfl⟩, rfl⟩
    /-
      case mk.mp.intro.mk.intro.up.up.intro.intro.intro.intro.intro
      X : AlgebraicGeometry.Scheme
      U : X.AffineZariskiSite
      s : Set ↑(X.presheaf.obj { unop := U.toOpens })
      V : X.Opens
      hV : AlgebraicGeometry.IsAffineOpen V
      f₂ : ↑(X.presheaf.obj { unop := U.toOpens })
      hf₂s : Membership.mem s f₂
      hW : AlgebraicGeometry.IsAffineOpen (X.basicOpen f₂)
      f₁ : ↑(X.presheaf.obj { unop := AlgebraicGeometry.Scheme.AffineZariskiSite.toO …
      hf₁ : Eq (X.basicOpen f₁) (AlgebraicGeometry.Scheme.AffineZariskiSite.toOpens  …
      ⊢ Exists fun f => And (Membership.mem s f) (Exists fun g => Eq (X.basicOpen (H …
    -/
    subst hf₁
    /-
      case mk.mp.intro.mk.intro.up.up.intro.intro.intro.intro.intro
      X : AlgebraicGeometry.Scheme
      U : X.AffineZariskiSite
      s : Set ↑(X.presheaf.obj { unop := U.toOpens })
      f₂ : ↑(X.presheaf.obj { unop := U.toOpens })
      hf₂s : Membership.mem s f₂
      hW : AlgebraicGeometry.IsAffineOpen (X.basicOpen f₂)
      f₁ : ↑(X.presheaf.obj { unop := AlgebraicGeometry.Scheme.AffineZariskiSite.toO …
      hV : AlgebraicGeometry.IsAffineOpen (X.basicOpen f₁)
      ⊢ Exists fun f => And (Membership.mem s f) (Exists fun g => Eq (X.basicOpen (H …
    -/
    obtain ⟨f₃, hf₃⟩ := U.2.basicOpen_basicOpen_is_basicOpen f₂ f₁
    /-
      case mk.mp.intro.mk.intro.up.up.intro.intro.intro.intro.intro.intro
      X : AlgebraicGeometry.Scheme
      U : X.AffineZariskiSite
      s : Set ↑(X.presheaf.obj { unop := U.toOpens })
      f₂ : ↑(X.presheaf.obj { unop := U.toOpens })
      hf₂s : Membership.mem s f₂
      hW : AlgebraicGeometry.IsAffineOpen (X.basicOpen f₂)
      f₁ : ↑(X.presheaf.obj { unop := AlgebraicGeometry.Scheme.AffineZariskiSite.toO …
      hV : AlgebraicGeometry.IsAffineOpen (X.basicOpen f₁)
      f₃ : ↑(X.presheaf.obj { unop := ↑U })
      hf₃ : Eq (X.basicOpen f₃) (X.basicOpen f₁)
      ⊢ Exists fun f => And (Membership.mem s f) (Exists fun g => Eq (X.basicOpen (H …
    -/
    refine ⟨f₂, hf₂s, f₃, ?_⟩
    /-
      case mk.mp.intro.mk.intro.up.up.intro.intro.intro.intro.intro.intro
      X : AlgebraicGeometry.Scheme
      U : X.AffineZariskiSite
      s : Set ↑(X.presheaf.obj { unop := U.toOpens })
      f₂ : ↑(X.presheaf.obj { unop := U.toOpens })
      hf₂s : Membership.mem s f₂
      hW : AlgebraicGeometry.IsAffineOpen (X.basicOpen f₂)
      f₁ : ↑(X.presheaf.obj { unop := AlgebraicGeometry.Scheme.AffineZariskiSite.toO …
      hV : AlgebraicGeometry.IsAffineOpen (X.basicOpen f₁)
      f₃ : ↑(X.presheaf.obj { unop := ↑U })
      hf₃ : Eq (X.basicOpen f₃) (X.basicOpen f₁)
      ⊢ Eq (X.basicOpen (HMul.hMul f₂ f₃)) (AlgebraicGeometry.Scheme.AffineZariskiSi …
    -/
    rw [X.basicOpen_mul, hf₃, inf_eq_right]
    /-
      case mk.mp.intro.mk.intro.up.up.intro.intro.intro.intro.intro.intro
      X : AlgebraicGeometry.Scheme
      U : X.AffineZariskiSite
      s : Set ↑(X.presheaf.obj { unop := U.toOpens })
      f₂ : ↑(X.presheaf.obj { unop := U.toOpens })
      hf₂s : Membership.mem s f₂
      hW : AlgebraicGeometry.IsAffineOpen (X.basicOpen f₂)
      f₁ : ↑(X.presheaf.obj { unop := AlgebraicGeometry.Scheme.AffineZariskiSite.toO …
      hV : AlgebraicGeometry.IsAffineOpen (X.basicOpen f₁)
      f₃ : ↑(X.presheaf.obj { unop := ↑U })
      hf₃ : Eq (X.basicOpen f₃) (X.basicOpen f₁)
      ⊢ LE.le (X.basicOpen f₁) (X.basicOpen f₂)
    -/
    exact X.basicOpen_le _
    /-
      🎉 no goals
    -/
    /-
      case mk.mpr
      X : AlgebraicGeometry.Scheme
      U : X.AffineZariskiSite
      s : Set ↑(X.presheaf.obj { unop := U.toOpens })
      V : X.Opens
      hV : AlgebraicGeometry.IsAffineOpen V
      f : Quiver.Hom ⟨V, hV⟩ U
      ⊢ (Exists fun f => And (Membership.mem s f) (Exists fun g => Eq (X.basicOpen ( …
    -/
  · rintro ⟨f₁, hf₁s, f₂, rfl⟩
    /-
      case mk.mpr.intro.intro.intro
      X : AlgebraicGeometry.Scheme
      U : X.AffineZariskiSite
      s : Set ↑(X.presheaf.obj { unop := U.toOpens })
      f₁ : ↑(X.presheaf.obj { unop := U.toOpens })
      hf₁s : Membership.mem s f₁
      f₂ : ↑(X.presheaf.obj { unop := U.toOpens })
      hV : AlgebraicGeometry.IsAffineOpen (X.basicOpen (HMul.hMul f₁ f₂))
      f : Quiver.Hom ⟨X.basicOpen (HMul.hMul f₁ f₂), hV⟩ U
      ⊢ (CategoryTheory.Sieve.generate (U.presieveOfSections s)).arrows f
    -/
    refine ⟨U.basicOpen f₁, ⟨f₂ |_ᵣ _, ?_⟩, ⟨f₁, rfl⟩, ⟨f₁, hf₁s, rfl⟩, rfl⟩
    /-
      case mk.mpr.intro.intro.intro
      X : AlgebraicGeometry.Scheme
      U : X.AffineZariskiSite
      s : Set ↑(X.presheaf.obj { unop := U.toOpens })
      f₁ : ↑(X.presheaf.obj { unop := U.toOpens })
      hf₁s : Membership.mem s f₁
      f₂ : ↑(X.presheaf.obj { unop := U.toOpens })
      hV : AlgebraicGeometry.IsAffineOpen (X.basicOpen (HMul.hMul f₁ f₂))
      f : Quiver.Hom ⟨X.basicOpen (HMul.hMul f₁ f₂), hV⟩ U
      ⊢ Eq (X.basicOpen (TopCat.Presheaf.restrictOpenCommRingCat f₂ (U.basicOpen f₁) …
    -/
    exact (X.basicOpen_res _ _).trans (X.basicOpen_mul _ _).symm
    /-
      🎉 no goals
    -/


lemma generate_presieveOfSections_mem_grothendieckTopology
    {U : X.AffineZariskiSite} {s : Set Γ(X, U.toOpens)} :
    Sieve.generate (presieveOfSections U s) ∈ grothendieckTopology X U ↔ Ideal.span s = ⊤ := by
  /-
    X : AlgebraicGeometry.Scheme
    U : X.AffineZariskiSite
    s : Set ↑(X.presheaf.obj { unop := U.toOpens })
    ⊢ Iff (Membership.mem ((AlgebraicGeometry.Scheme.AffineZariskiSite.grothendiec …
  -/
  rw [← U.2.self_le_basicOpen_union_iff, mem_grothendieckTopology, SetLike.le_def]
  /-
    X : AlgebraicGeometry.Scheme
    U : X.AffineZariskiSite
    s : Set ↑(X.presheaf.obj { unop := U.toOpens })
    ⊢ Iff (∀ (x : ↑↑X.toPresheafedSpace), Membership.mem U.toOpens x → Exists fun  …
  -/
  refine forall₂_congr fun x hx ↦ ?_
  simp only [exists_and_left, TopologicalSpace.Opens.iSup_mk,
    TopologicalSpace.Opens.carrier_eq_coe, Set.iUnion_coe_set, TopologicalSpace.Opens.mem_mk,
    Set.mem_iUnion, SetLike.mem_coe, exists_prop, generate_presieveOfSections]
  /-
    X : AlgebraicGeometry.Scheme
    U : X.AffineZariskiSite
    s : Set ↑(X.presheaf.obj { unop := U.toOpens })
    x : ↑↑X.toPresheafedSpace
    hx : Membership.mem U.toOpens x
    ⊢ Iff (Exists fun V => And (Exists fun f => And (Membership.mem s f) (Exists f …
  -/
  constructor
    /-
      case mp
      X : AlgebraicGeometry.Scheme
      U : X.AffineZariskiSite
      s : Set ↑(X.presheaf.obj { unop := U.toOpens })
      x : ↑↑X.toPresheafedSpace
      hx : Membership.mem U.toOpens x
      ⊢ (Exists fun V => And (Exists fun f => And (Membership.mem s f) (Exists fun g …
    -/
  · simp only [basicOpen_mul]
    /-
      case mp
      X : AlgebraicGeometry.Scheme
      U : X.AffineZariskiSite
      s : Set ↑(X.presheaf.obj { unop := U.toOpens })
      x : ↑↑X.toPresheafedSpace
      hx : Membership.mem U.toOpens x
      ⊢ (Exists fun V => And (Exists fun f => And (Membership.mem s f) (Exists fun g …
    -/
    rintro ⟨⟨V, hV⟩, ⟨f, hfs, g, rfl⟩, -, hxV⟩
    /-
      case mp.intro.mk.intro.intro.intro.intro.intro
      X : AlgebraicGeometry.Scheme
      U : X.AffineZariskiSite
      s : Set ↑(X.presheaf.obj { unop := U.toOpens })
      x : ↑↑X.toPresheafedSpace
      hx : Membership.mem U.toOpens x
      f : ↑(X.presheaf.obj { unop := U.toOpens })
      hfs : Membership.mem s f
      g : ↑(X.presheaf.obj { unop := U.toOpens })
      hV : AlgebraicGeometry.IsAffineOpen (Min.min (X.basicOpen f) (X.basicOpen g))
      hxV : Membership.mem (AlgebraicGeometry.Scheme.AffineZariskiSite.toOpens ⟨Min. …
      ⊢ Exists fun i => And (Membership.mem s i) (Membership.mem (X.basicOpen i) x)
    -/
    exact ⟨f, hfs, hxV.1⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      X : AlgebraicGeometry.Scheme
      U : X.AffineZariskiSite
      s : Set ↑(X.presheaf.obj { unop := U.toOpens })
      x : ↑↑X.toPresheafedSpace
      hx : Membership.mem U.toOpens x
      ⊢ (Exists fun i => And (Membership.mem s i) (Membership.mem (X.basicOpen i) x) …
    -/
  · rintro ⟨f, hfs, hxf⟩
    /-
      case mpr.intro.intro
      X : AlgebraicGeometry.Scheme
      U : X.AffineZariskiSite
      s : Set ↑(X.presheaf.obj { unop := U.toOpens })
      x : ↑↑X.toPresheafedSpace
      hx : Membership.mem U.toOpens x
      f : ↑(X.presheaf.obj { unop := ↑U })
      hfs : Membership.mem s f
      hxf : Membership.mem (X.basicOpen f) x
      ⊢ Exists fun V => And (Exists fun f => And (Membership.mem s f) (Exists fun g  …
    -/
    refine ⟨U.basicOpen _, ⟨f, hfs, 1, rfl⟩, ⟨_, rfl⟩, by simpa using hxf⟩
    /-
      🎉 no goals
    -/


lemma mem_grothendieckTopology_iff_sectionsOfPresieve
    {U : X.AffineZariskiSite} {S : Sieve U} :
    S ∈ grothendieckTopology X U ↔ Ideal.span (sectionsOfPresieve S.1) = ⊤ := by
  rw [← generate_presieveOfSections_mem_grothendieckTopology, presieveOfSections_sectionsOfPresieve,
    Sieve.generate_sieve]


/-- The category of sheaves on `X.AffineZariskiSite` is equivalent to the categories of sheaves
over `X`. -/
abbrev sheafEquiv : Sheaf (grothendieckTopology X) A ≌ TopCat.Sheaf A X :=
    (toOpensFunctor X).sheafInducedTopologyEquivOfIsCoverDense _ _


