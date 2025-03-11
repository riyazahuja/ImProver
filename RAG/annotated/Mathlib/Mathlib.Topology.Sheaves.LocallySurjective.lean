/-- A map of presheaves `T : ℱ ⟶ 𝒢` is **locally surjective** if for any open set `U`,
section `t` over `U`, and `x ∈ U`, there exists an open set `x ∈ V ⊆ U` and a section `s` over `V`
such that `$T_*(s_V) = t|_V$`.

See `TopCat.Presheaf.isLocallySurjective_iff` below.
-/
def IsLocallySurjective (T : ℱ ⟶ 𝒢) :=
  CategoryTheory.Presheaf.IsLocallySurjective (Opens.grothendieckTopology X) T


theorem isLocallySurjective_iff (T : ℱ ⟶ 𝒢) :
    IsLocallySurjective T ↔
      ∀ (U t), ∀ x ∈ U, ∃ (V : _) (ι : V ⟶ U), (∃ s, T.app _ s = t |_ₕ ι) ∧ x ∈ V :=
  ⟨fun h _ => h.imageSieve_mem, fun h => ⟨h _⟩⟩


/-- An equivalent condition for a map of presheaves to be locally surjective
is for all the induced maps on stalks to be surjective. -/
theorem locally_surjective_iff_surjective_on_stalks (T : ℱ ⟶ 𝒢) :
    IsLocallySurjective T ↔ ∀ x : X, Function.Surjective ((stalkFunctor C x).map T) := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.ConcreteCategory C
    X : TopCat
    ℱ 𝒢 : TopCat.Presheaf C X
    inst✝¹ : CategoryTheory.Limits.HasColimits C
    inst✝ : CategoryTheory.Limits.PreservesFilteredColimits (CategoryTheory.forget …
    T : Quiver.Hom ℱ 𝒢
    ⊢ Iff (TopCat.Presheaf.IsLocallySurjective T) (∀ (x : ↑X), Function.Surjective …
  -/
  constructor <;> intro hT
  · /- human proof:
        Let g ∈ Γₛₜ 𝒢 x be a germ. Represent it on an open set U ⊆ X
        as ⟨t, U⟩. By local surjectivity, pass to a smaller open set V
        on which there exists s ∈ Γ_ ℱ V mapping to t |_ V.
        Then the germ of s maps to g -/
    -- Let g ∈ Γₛₜ 𝒢 x be a germ.
    /-
      case mp
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.ConcreteCategory C
      X : TopCat
      ℱ 𝒢 : TopCat.Presheaf C X
      inst✝¹ : CategoryTheory.Limits.HasColimits C
      inst✝ : CategoryTheory.Limits.PreservesFilteredColimits (CategoryTheory.forget …
      T : Quiver.Hom ℱ 𝒢
      hT : TopCat.Presheaf.IsLocallySurjective T
      ⊢ ∀ (x : ↑X), Function.Surjective ⇑((TopCat.Presheaf.stalkFunctor C x).map T)
    -/
    intro x g
    -- Represent it on an open set U ⊆ X as ⟨t, U⟩.
    /-
      case mp
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.ConcreteCategory C
      X : TopCat
      ℱ 𝒢 : TopCat.Presheaf C X
      inst✝¹ : CategoryTheory.Limits.HasColimits C
      inst✝ : CategoryTheory.Limits.PreservesFilteredColimits (CategoryTheory.forget …
      T : Quiver.Hom ℱ 𝒢
      hT : TopCat.Presheaf.IsLocallySurjective T
      x : ↑X
      g : (CategoryTheory.forget C).obj ((TopCat.Presheaf.stalkFunctor C x).obj 𝒢)
      ⊢ Exists fun a => Eq (((TopCat.Presheaf.stalkFunctor C x).map T) a) g
    -/
    obtain ⟨U, hxU, t, rfl⟩ := 𝒢.germ_exist x g
    -- By local surjectivity, pass to a smaller open set V
    -- on which there exists s ∈ Γ_ ℱ V mapping to t |_ V.
    /-
      case mp.intro.intro.intro
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.ConcreteCategory C
      X : TopCat
      ℱ 𝒢 : TopCat.Presheaf C X
      inst✝¹ : CategoryTheory.Limits.HasColimits C
      inst✝ : CategoryTheory.Limits.PreservesFilteredColimits (CategoryTheory.forget …
      T : Quiver.Hom ℱ 𝒢
      hT : TopCat.Presheaf.IsLocallySurjective T
      x : ↑X
      U : TopologicalSpace.Opens ↑X
      hxU : Membership.mem U x
      t : (CategoryTheory.forget C).obj (𝒢.obj { unop := U })
      ⊢ Exists fun a => Eq (((TopCat.Presheaf.stalkFunctor C x).map T) a) ((𝒢.germ U …
    -/
    rcases hT.imageSieve_mem t x hxU with ⟨V, ι, ⟨s, h_eq⟩, hxV⟩
    -- Then the germ of s maps to g.
    /-
      case mp.intro.intro.intro.intro.intro.intro.intro
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.ConcreteCategory C
      X : TopCat
      ℱ 𝒢 : TopCat.Presheaf C X
      inst✝¹ : CategoryTheory.Limits.HasColimits C
      inst✝ : CategoryTheory.Limits.PreservesFilteredColimits (CategoryTheory.forget …
      T : Quiver.Hom ℱ 𝒢
      hT : TopCat.Presheaf.IsLocallySurjective T
      x : ↑X
      U : TopologicalSpace.Opens ↑X
      hxU : Membership.mem U x
      t : (CategoryTheory.forget C).obj (𝒢.obj { unop := U })
      V : TopologicalSpace.Opens ↑X
      ι : Quiver.Hom V U
      hxV : Membership.mem V x
      s : (CategoryTheory.forget C).obj (ℱ.obj { unop := V })
      h_eq : Eq ((T.app { unop := V }) s) ((𝒢.map ι.op) t)
      ⊢ Exists fun a => Eq (((TopCat.Presheaf.stalkFunctor C x).map T) a) ((𝒢.germ U …
    -/
    use ℱ.germ _ x hxV s
    -- Porting note: `convert` went too deep and swapped LHS and RHS of the remaining goal relative
    -- to lean 3.
    /-
      case h
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.ConcreteCategory C
      X : TopCat
      ℱ 𝒢 : TopCat.Presheaf C X
      inst✝¹ : CategoryTheory.Limits.HasColimits C
      inst✝ : CategoryTheory.Limits.PreservesFilteredColimits (CategoryTheory.forget …
      T : Quiver.Hom ℱ 𝒢
      hT : TopCat.Presheaf.IsLocallySurjective T
      x : ↑X
      U : TopologicalSpace.Opens ↑X
      hxU : Membership.mem U x
      t : (CategoryTheory.forget C).obj (𝒢.obj { unop := U })
      V : TopologicalSpace.Opens ↑X
      ι : Quiver.Hom V U
      hxV : Membership.mem V x
      s : (CategoryTheory.forget C).obj (ℱ.obj { unop := V })
      h_eq : Eq ((T.app { unop := V }) s) ((𝒢.map ι.op) t)
      ⊢ Eq (((TopCat.Presheaf.stalkFunctor C x).map T) ((ℱ.germ V x hxV) s)) ((𝒢.ger …
    -/
    convert stalkFunctor_map_germ_apply V x hxV T s using 1
    /-
      case h.e'_3
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.ConcreteCategory C
      X : TopCat
      ℱ 𝒢 : TopCat.Presheaf C X
      inst✝¹ : CategoryTheory.Limits.HasColimits C
      inst✝ : CategoryTheory.Limits.PreservesFilteredColimits (CategoryTheory.forget …
      T : Quiver.Hom ℱ 𝒢
      hT : TopCat.Presheaf.IsLocallySurjective T
      x : ↑X
      U : TopologicalSpace.Opens ↑X
      hxU : Membership.mem U x
      t : (CategoryTheory.forget C).obj (𝒢.obj { unop := U })
      V : TopologicalSpace.Opens ↑X
      ι : Quiver.Hom V U
      hxV : Membership.mem V x
      s : (CategoryTheory.forget C).obj (ℱ.obj { unop := V })
      h_eq : Eq ((T.app { unop := V }) s) ((𝒢.map ι.op) t)
      ⊢ Eq ((𝒢.germ U x hxU) t) ((𝒢.germ V x hxV) ((T.app { unop := V }) s))
    -/
    simpa [h_eq] using (germ_res_apply 𝒢 ι x hxV t).symm
    /-
      🎉 no goals
    -/
  · /- human proof:
        Let U be an open set, t ∈ Γ ℱ U a section, x ∈ U a point.
        By surjectivity on stalks, the germ of t is the image of
        some germ f ∈ Γₛₜ ℱ x. Represent f on some open set V ⊆ X as ⟨s, V⟩.
        Then there is some possibly smaller open set x ∈ W ⊆ V ∩ U on which
        we have T(s) |_ W = t |_ W. -/
    /-
      case mpr
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.ConcreteCategory C
      X : TopCat
      ℱ 𝒢 : TopCat.Presheaf C X
      inst✝¹ : CategoryTheory.Limits.HasColimits C
      inst✝ : CategoryTheory.Limits.PreservesFilteredColimits (CategoryTheory.forget …
      T : Quiver.Hom ℱ 𝒢
      hT : ∀ (x : ↑X), Function.Surjective ⇑((TopCat.Presheaf.stalkFunctor C x).map T)
      ⊢ TopCat.Presheaf.IsLocallySurjective T
    -/
    constructor
    /-
      case mpr.imageSieve_mem
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.ConcreteCategory C
      X : TopCat
      ℱ 𝒢 : TopCat.Presheaf C X
      inst✝¹ : CategoryTheory.Limits.HasColimits C
      inst✝ : CategoryTheory.Limits.PreservesFilteredColimits (CategoryTheory.forget …
      T : Quiver.Hom ℱ 𝒢
      hT : ∀ (x : ↑X), Function.Surjective ⇑((TopCat.Presheaf.stalkFunctor C x).map T)
      ⊢ ∀ {U : TopologicalSpace.Opens ↑X} (s : (CategoryTheory.forget C).obj (𝒢.obj  …
    -/
    intro U t x hxU
    /-
      case mpr.imageSieve_mem
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.ConcreteCategory C
      X : TopCat
      ℱ 𝒢 : TopCat.Presheaf C X
      inst✝¹ : CategoryTheory.Limits.HasColimits C
      inst✝ : CategoryTheory.Limits.PreservesFilteredColimits (CategoryTheory.forget …
      T : Quiver.Hom ℱ 𝒢
      hT : ∀ (x : ↑X), Function.Surjective ⇑((TopCat.Presheaf.stalkFunctor C x).map T)
      U : TopologicalSpace.Opens ↑X
      t : (CategoryTheory.forget C).obj (𝒢.obj { unop := U })
      x : ↑X
      hxU : Membership.mem U x
      ⊢ Exists fun U_1 => Exists fun f => And ((CategoryTheory.Presheaf.imageSieve T …
    -/
    set t_x := 𝒢.germ _ x hxU t with ht_x
    /-
      case mpr.imageSieve_mem
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.ConcreteCategory C
      X : TopCat
      ℱ 𝒢 : TopCat.Presheaf C X
      inst✝¹ : CategoryTheory.Limits.HasColimits C
      inst✝ : CategoryTheory.Limits.PreservesFilteredColimits (CategoryTheory.forget …
      T : Quiver.Hom ℱ 𝒢
      hT : ∀ (x : ↑X), Function.Surjective ⇑((TopCat.Presheaf.stalkFunctor C x).map T)
      U : TopologicalSpace.Opens ↑X
      t : (CategoryTheory.forget C).obj (𝒢.obj { unop := U })
      x : ↑X
      hxU : Membership.mem U x
      t_x : (CategoryTheory.forget C).obj (𝒢.stalk x) := (𝒢.germ U x hxU) t
      ht_x : Eq t_x ((𝒢.germ U x hxU) t)
      ⊢ Exists fun U_1 => Exists fun f => And ((CategoryTheory.Presheaf.imageSieve T …
    -/
    obtain ⟨s_x, hs_x : ((stalkFunctor C x).map T) s_x = t_x⟩ := hT x t_x
    /-
      case mpr.imageSieve_mem.intro
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.ConcreteCategory C
      X : TopCat
      ℱ 𝒢 : TopCat.Presheaf C X
      inst✝¹ : CategoryTheory.Limits.HasColimits C
      inst✝ : CategoryTheory.Limits.PreservesFilteredColimits (CategoryTheory.forget …
      T : Quiver.Hom ℱ 𝒢
      hT : ∀ (x : ↑X), Function.Surjective ⇑((TopCat.Presheaf.stalkFunctor C x).map T)
      U : TopologicalSpace.Opens ↑X
      t : (CategoryTheory.forget C).obj (𝒢.obj { unop := U })
      x : ↑X
      hxU : Membership.mem U x
      t_x : (CategoryTheory.forget C).obj (𝒢.stalk x) := (𝒢.germ U x hxU) t
      ht_x : Eq t_x ((𝒢.germ U x hxU) t)
      s_x : (CategoryTheory.forget C).obj ((TopCat.Presheaf.stalkFunctor C x).obj ℱ)
      hs_x : Eq (((TopCat.Presheaf.stalkFunctor C x).map T) s_x) t_x
      ⊢ Exists fun U_1 => Exists fun f => And ((CategoryTheory.Presheaf.imageSieve T …
    -/
    obtain ⟨V, hxV, s, rfl⟩ := ℱ.germ_exist x s_x
    -- rfl : ℱ.germ x s = s_x
    have key_W := 𝒢.germ_eq x hxV hxU (T.app _ s) t <| by
      convert hs_x using 1
      symm
      convert stalkFunctor_map_germ_apply _ _ _ _ s
    /-
      case mpr.imageSieve_mem.intro.intro.intro.intro
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.ConcreteCategory C
      X : TopCat
      ℱ 𝒢 : TopCat.Presheaf C X
      inst✝¹ : CategoryTheory.Limits.HasColimits C
      inst✝ : CategoryTheory.Limits.PreservesFilteredColimits (CategoryTheory.forget …
      T : Quiver.Hom ℱ 𝒢
      hT : ∀ (x : ↑X), Function.Surjective ⇑((TopCat.Presheaf.stalkFunctor C x).map T)
      U : TopologicalSpace.Opens ↑X
      t : (CategoryTheory.forget C).obj (𝒢.obj { unop := U })
      x : ↑X
      hxU : Membership.mem U x
      t_x : (CategoryTheory.forget C).obj (𝒢.stalk x) := (𝒢.germ U x hxU) t
      ht_x : Eq t_x ((𝒢.germ U x hxU) t)
      V : TopologicalSpace.Opens ↑X
      hxV : Membership.mem V x
      s : (CategoryTheory.forget C).obj (ℱ.obj { unop := V })
      hs_x : Eq (((TopCat.Presheaf.stalkFunctor C x).map T) ((ℱ.germ V x hxV) s)) t_x
      key_W : Exists fun W => Exists fun _m => Exists fun iU => Exists fun iV => Eq  …
      ⊢ Exists fun U_1 => Exists fun f => And ((CategoryTheory.Presheaf.imageSieve T …
    -/
    obtain ⟨W, hxW, hWV, hWU, h_eq⟩ := key_W
    /-
      case mpr.imageSieve_mem.intro.intro.intro.intro.intro.intro.intro.intro
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.ConcreteCategory C
      X : TopCat
      ℱ 𝒢 : TopCat.Presheaf C X
      inst✝¹ : CategoryTheory.Limits.HasColimits C
      inst✝ : CategoryTheory.Limits.PreservesFilteredColimits (CategoryTheory.forget …
      T : Quiver.Hom ℱ 𝒢
      hT : ∀ (x : ↑X), Function.Surjective ⇑((TopCat.Presheaf.stalkFunctor C x).map T)
      U : TopologicalSpace.Opens ↑X
      t : (CategoryTheory.forget C).obj (𝒢.obj { unop := U })
      x : ↑X
      hxU : Membership.mem U x
      t_x : (CategoryTheory.forget C).obj (𝒢.stalk x) := (𝒢.germ U x hxU) t
      ht_x : Eq t_x ((𝒢.germ U x hxU) t)
      V : TopologicalSpace.Opens ↑X
      hxV : Membership.mem V x
      s : (CategoryTheory.forget C).obj (ℱ.obj { unop := V })
      hs_x : Eq (((TopCat.Presheaf.stalkFunctor C x).map T) ((ℱ.germ V x hxV) s)) t_x
      W : TopologicalSpace.Opens ↑X
      hxW : Membership.mem W x
      hWV : Quiver.Hom W V
      hWU : Quiver.Hom W U
      h_eq : Eq ((𝒢.map hWV.op) ((T.app { unop := V }) s)) ((𝒢.map hWU.op) t)
      ⊢ Exists fun U_1 => Exists fun f => And ((CategoryTheory.Presheaf.imageSieve T …
    -/
    refine ⟨W, hWU, ⟨ℱ.map hWV.op s, ?_⟩, hxW⟩
    /-
      case mpr.imageSieve_mem.intro.intro.intro.intro.intro.intro.intro.intro
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.ConcreteCategory C
      X : TopCat
      ℱ 𝒢 : TopCat.Presheaf C X
      inst✝¹ : CategoryTheory.Limits.HasColimits C
      inst✝ : CategoryTheory.Limits.PreservesFilteredColimits (CategoryTheory.forget …
      T : Quiver.Hom ℱ 𝒢
      hT : ∀ (x : ↑X), Function.Surjective ⇑((TopCat.Presheaf.stalkFunctor C x).map T)
      U : TopologicalSpace.Opens ↑X
      t : (CategoryTheory.forget C).obj (𝒢.obj { unop := U })
      x : ↑X
      hxU : Membership.mem U x
      t_x : (CategoryTheory.forget C).obj (𝒢.stalk x) := (𝒢.germ U x hxU) t
      ht_x : Eq t_x ((𝒢.germ U x hxU) t)
      V : TopologicalSpace.Opens ↑X
      hxV : Membership.mem V x
      s : (CategoryTheory.forget C).obj (ℱ.obj { unop := V })
      hs_x : Eq (((TopCat.Presheaf.stalkFunctor C x).map T) ((ℱ.germ V x hxV) s)) t_x
      W : TopologicalSpace.Opens ↑X
      hxW : Membership.mem W x
      hWV : Quiver.Hom W V
      hWU : Quiver.Hom W U
      h_eq : Eq ((𝒢.map hWV.op) ((T.app { unop := V }) s)) ((𝒢.map hWU.op) t)
      ⊢ Eq ((T.app { unop := W }) ((ℱ.map hWV.op) s)) ((𝒢.map hWU.op) t)
    -/
    convert h_eq using 1
    /-
      case h.e'_2
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.ConcreteCategory C
      X : TopCat
      ℱ 𝒢 : TopCat.Presheaf C X
      inst✝¹ : CategoryTheory.Limits.HasColimits C
      inst✝ : CategoryTheory.Limits.PreservesFilteredColimits (CategoryTheory.forget …
      T : Quiver.Hom ℱ 𝒢
      hT : ∀ (x : ↑X), Function.Surjective ⇑((TopCat.Presheaf.stalkFunctor C x).map T)
      U : TopologicalSpace.Opens ↑X
      t : (CategoryTheory.forget C).obj (𝒢.obj { unop := U })
      x : ↑X
      hxU : Membership.mem U x
      t_x : (CategoryTheory.forget C).obj (𝒢.stalk x) := (𝒢.germ U x hxU) t
      ht_x : Eq t_x ((𝒢.germ U x hxU) t)
      V : TopologicalSpace.Opens ↑X
      hxV : Membership.mem V x
      s : (CategoryTheory.forget C).obj (ℱ.obj { unop := V })
      hs_x : Eq (((TopCat.Presheaf.stalkFunctor C x).map T) ((ℱ.germ V x hxV) s)) t_x
      W : TopologicalSpace.Opens ↑X
      hxW : Membership.mem W x
      hWV : Quiver.Hom W V
      hWU : Quiver.Hom W U
      h_eq : Eq ((𝒢.map hWV.op) ((T.app { unop := V }) s)) ((𝒢.map hWU.op) t)
      ⊢ Eq ((T.app { unop := W }) ((ℱ.map hWV.op) s)) ((𝒢.map hWV.op) ((T.app { unop …
    -/
    simp only [← comp_apply, T.naturality]
    /-
      🎉 no goals
    -/


