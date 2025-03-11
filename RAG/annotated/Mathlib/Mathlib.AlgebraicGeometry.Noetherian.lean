/-- A scheme `X` is locally Noetherian if `𝒪ₓ(U)` is Noetherian for all affine `U`. -/
class IsLocallyNoetherian (X : Scheme) : Prop where
  component_noetherian : ∀ (U : X.affineOpens),
    IsNoetherianRing Γ(X, U) := by infer_instance


include hS hN in
/-- Let `R` be a ring, and `f i` a finite collection of elements of `R` generating the unit ideal.
If the localization of `R` at each `f i` is noetherian, so is `R`.

We follow the proof given in [Har77], Proposition II.3.2 -/
theorem isNoetherianRing_of_away : IsNoetherianRing R := by
  /-
    R : Type u
    inst✝ : CommRing R
    S : Finset R
    hS : Eq (Ideal.span ↑S) Top.top
    hN : ∀ (s : Subtype fun x => Membership.mem S x), IsNoetherianRing (Localizati …
    ⊢ IsNoetherianRing R
  -/
  apply monotone_stabilizes_iff_noetherian.mp
  /-
    R : Type u
    inst✝ : CommRing R
    S : Finset R
    hS : Eq (Ideal.span ↑S) Top.top
    hN : ∀ (s : Subtype fun x => Membership.mem S x), IsNoetherianRing (Localizati …
    ⊢ ∀ (f : OrderHom Nat (Submodule R R)), Exists fun n => ∀ (m : Nat), LE.le n m …
  -/
  intro I
  /-
    R : Type u
    inst✝ : CommRing R
    S : Finset R
    hS : Eq (Ideal.span ↑S) Top.top
    hN : ∀ (s : Subtype fun x => Membership.mem S x), IsNoetherianRing (Localizati …
    I : OrderHom Nat (Submodule R R)
    ⊢ Exists fun n => ∀ (m : Nat), LE.le n m → Eq (I n) (I m)
  -/
  let floc s := algebraMap R (Away (M := R) s)
  let suitableN s :=
    { n : ℕ | ∀ m : ℕ, n ≤ m → (Ideal.map (floc s) (I n)) = (Ideal.map (floc s) (I m)) }
  /-
    R : Type u
    inst✝ : CommRing R
    S : Finset R
    hS : Eq (Ideal.span ↑S) Top.top
    hN : ∀ (s : Subtype fun x => Membership.mem S x), IsNoetherianRing (Localizati …
    I : OrderHom Nat (Submodule R R)
    floc : (s : R) → RingHom R (Localization.Away s) := fun s => algebraMap R (Loc …
    suitableN : R → Set Nat := fun s => setOf fun n => ∀ (m : Nat), LE.le n m → Eq …
    ⊢ Exists fun n => ∀ (m : Nat), LE.le n m → Eq (I n) (I m)
  -/
  let minN s := sInf (suitableN s)
  have hSuit : ∀ s : S, minN s ∈ suitableN s := by
    intro s
    apply Nat.sInf_mem
    let f : ℕ →o Ideal (Away (M := R) s) :=
      ⟨fun n ↦ Ideal.map (floc s) (I n), fun _ _ h ↦ Ideal.map_mono (I.monotone h)⟩
    exact monotone_stabilizes_iff_noetherian.mpr (hN s) f
  /-
    R : Type u
    inst✝ : CommRing R
    S : Finset R
    hS : Eq (Ideal.span ↑S) Top.top
    hN : ∀ (s : Subtype fun x => Membership.mem S x), IsNoetherianRing (Localizati …
    I : OrderHom Nat (Submodule R R)
    floc : (s : R) → RingHom R (Localization.Away s) := fun s => algebraMap R (Loc …
    suitableN : R → Set Nat := fun s => setOf fun n => ∀ (m : Nat), LE.le n m → Eq …
    minN : R → Nat := fun s => InfSet.sInf (suitableN s)
    hSuit : ∀ (s : Subtype fun x => Membership.mem S x), Membership.mem (suitableN …
    ⊢ Exists fun n => ∀ (m : Nat), LE.le n m → Eq (I n) (I m)
  -/
  let N := Finset.sup S minN
  /-
    R : Type u
    inst✝ : CommRing R
    S : Finset R
    hS : Eq (Ideal.span ↑S) Top.top
    hN : ∀ (s : Subtype fun x => Membership.mem S x), IsNoetherianRing (Localizati …
    I : OrderHom Nat (Submodule R R)
    floc : (s : R) → RingHom R (Localization.Away s) := fun s => algebraMap R (Loc …
    suitableN : R → Set Nat := fun s => setOf fun n => ∀ (m : Nat), LE.le n m → Eq …
    minN : R → Nat := fun s => InfSet.sInf (suitableN s)
    hSuit : ∀ (s : Subtype fun x => Membership.mem S x), Membership.mem (suitableN …
    N : Nat := S.sup minN
    ⊢ Exists fun n => ∀ (m : Nat), LE.le n m → Eq (I n) (I m)
  -/
  use N
  /-
    case h
    R : Type u
    inst✝ : CommRing R
    S : Finset R
    hS : Eq (Ideal.span ↑S) Top.top
    hN : ∀ (s : Subtype fun x => Membership.mem S x), IsNoetherianRing (Localizati …
    I : OrderHom Nat (Submodule R R)
    floc : (s : R) → RingHom R (Localization.Away s) := fun s => algebraMap R (Loc …
    suitableN : R → Set Nat := fun s => setOf fun n => ∀ (m : Nat), LE.le n m → Eq …
    minN : R → Nat := fun s => InfSet.sInf (suitableN s)
    hSuit : ∀ (s : Subtype fun x => Membership.mem S x), Membership.mem (suitableN …
    N : Nat := S.sup minN
    ⊢ ∀ (m : Nat), LE.le N m → Eq (I N) (I m)
  -/
  have hN : ∀ s : S, minN s ≤ N := fun s => Finset.le_sup s.prop
  /-
    case h
    R : Type u
    inst✝ : CommRing R
    S : Finset R
    hS : Eq (Ideal.span ↑S) Top.top
    hN✝ : ∀ (s : Subtype fun x => Membership.mem S x), IsNoetherianRing (Localizat …
    I : OrderHom Nat (Submodule R R)
    floc : (s : R) → RingHom R (Localization.Away s) := fun s => algebraMap R (Loc …
    suitableN : R → Set Nat := fun s => setOf fun n => ∀ (m : Nat), LE.le n m → Eq …
    minN : R → Nat := fun s => InfSet.sInf (suitableN s)
    hSuit : ∀ (s : Subtype fun x => Membership.mem S x), Membership.mem (suitableN …
    N : Nat := S.sup minN
    hN : ∀ (s : Subtype fun x => Membership.mem S x), LE.le (minN ↑s) N
    ⊢ ∀ (m : Nat), LE.le N m → Eq (I N) (I m)
  -/
  intro n hn
  rw [IsLocalization.ideal_eq_iInf_comap_map_away hS (I N),
      IsLocalization.ideal_eq_iInf_comap_map_away hS (I n),
      iInf_subtype', iInf_subtype']
  /-
    case h
    R : Type u
    inst✝ : CommRing R
    S : Finset R
    hS : Eq (Ideal.span ↑S) Top.top
    hN✝ : ∀ (s : Subtype fun x => Membership.mem S x), IsNoetherianRing (Localizat …
    I : OrderHom Nat (Submodule R R)
    floc : (s : R) → RingHom R (Localization.Away s) := fun s => algebraMap R (Loc …
    suitableN : R → Set Nat := fun s => setOf fun n => ∀ (m : Nat), LE.le n m → Eq …
    minN : R → Nat := fun s => InfSet.sInf (suitableN s)
    hSuit : ∀ (s : Subtype fun x => Membership.mem S x), Membership.mem (suitableN …
    N : Nat := S.sup minN
    hN : ∀ (s : Subtype fun x => Membership.mem S x), LE.le (minN ↑s) N
    n : Nat
    hn : LE.le N n
    ⊢ Eq (iInf fun x => Ideal.comap (algebraMap R (Localization.Away ↑x)) (Ideal.m …
  -/
  apply iInf_congr
  /-
    case h.h
    R : Type u
    inst✝ : CommRing R
    S : Finset R
    hS : Eq (Ideal.span ↑S) Top.top
    hN✝ : ∀ (s : Subtype fun x => Membership.mem S x), IsNoetherianRing (Localizat …
    I : OrderHom Nat (Submodule R R)
    floc : (s : R) → RingHom R (Localization.Away s) := fun s => algebraMap R (Loc …
    suitableN : R → Set Nat := fun s => setOf fun n => ∀ (m : Nat), LE.le n m → Eq …
    minN : R → Nat := fun s => InfSet.sInf (suitableN s)
    hSuit : ∀ (s : Subtype fun x => Membership.mem S x), Membership.mem (suitableN …
    N : Nat := S.sup minN
    hN : ∀ (s : Subtype fun x => Membership.mem S x), LE.le (minN ↑s) N
    n : Nat
    hn : LE.le N n
    ⊢ ∀ (i : Subtype (Membership.mem S)), Eq (Ideal.comap (algebraMap R (Localizat …
  -/
  intro s
  /-
    case h.h
    R : Type u
    inst✝ : CommRing R
    S : Finset R
    hS : Eq (Ideal.span ↑S) Top.top
    hN✝ : ∀ (s : Subtype fun x => Membership.mem S x), IsNoetherianRing (Localizat …
    I : OrderHom Nat (Submodule R R)
    floc : (s : R) → RingHom R (Localization.Away s) := fun s => algebraMap R (Loc …
    suitableN : R → Set Nat := fun s => setOf fun n => ∀ (m : Nat), LE.le n m → Eq …
    minN : R → Nat := fun s => InfSet.sInf (suitableN s)
    hSuit : ∀ (s : Subtype fun x => Membership.mem S x), Membership.mem (suitableN …
    N : Nat := S.sup minN
    hN : ∀ (s : Subtype fun x => Membership.mem S x), LE.le (minN ↑s) N
    n : Nat
    hn : LE.le N n
    s : Subtype (Membership.mem S)
    ⊢ Eq (Ideal.comap (algebraMap R (Localization.Away ↑s)) (Ideal.map (algebraMap …
  -/
  congr 1
  /-
    case h.h.e_I
    R : Type u
    inst✝ : CommRing R
    S : Finset R
    hS : Eq (Ideal.span ↑S) Top.top
    hN✝ : ∀ (s : Subtype fun x => Membership.mem S x), IsNoetherianRing (Localizat …
    I : OrderHom Nat (Submodule R R)
    floc : (s : R) → RingHom R (Localization.Away s) := fun s => algebraMap R (Loc …
    suitableN : R → Set Nat := fun s => setOf fun n => ∀ (m : Nat), LE.le n m → Eq …
    minN : R → Nat := fun s => InfSet.sInf (suitableN s)
    hSuit : ∀ (s : Subtype fun x => Membership.mem S x), Membership.mem (suitableN …
    N : Nat := S.sup minN
    hN : ∀ (s : Subtype fun x => Membership.mem S x), LE.le (minN ↑s) N
    n : Nat
    hn : LE.le N n
    s : Subtype (Membership.mem S)
    ⊢ Eq (Ideal.map (algebraMap R (Localization.Away ↑s)) (I N)) (Ideal.map (algeb …
  -/
  rw [← hSuit s N (hN s)]
  /-
    case h.h.e_I
    R : Type u
    inst✝ : CommRing R
    S : Finset R
    hS : Eq (Ideal.span ↑S) Top.top
    hN✝ : ∀ (s : Subtype fun x => Membership.mem S x), IsNoetherianRing (Localizat …
    I : OrderHom Nat (Submodule R R)
    floc : (s : R) → RingHom R (Localization.Away s) := fun s => algebraMap R (Loc …
    suitableN : R → Set Nat := fun s => setOf fun n => ∀ (m : Nat), LE.le n m → Eq …
    minN : R → Nat := fun s => InfSet.sInf (suitableN s)
    hSuit : ∀ (s : Subtype fun x => Membership.mem S x), Membership.mem (suitableN …
    N : Nat := S.sup minN
    hN : ∀ (s : Subtype fun x => Membership.mem S x), LE.le (minN ↑s) N
    n : Nat
    hn : LE.le N n
    s : Subtype (Membership.mem S)
    ⊢ Eq (Ideal.map (floc ↑s) (I (minN ↑s))) (Ideal.map (algebraMap R (Localizatio …
  -/
  exact hSuit s n <| Nat.le_trans (hN s) hn
  /-
    🎉 no goals
  -/


/-- If a scheme `X` has a cover by affine opens whose sections are Noetherian rings,
then `X` is locally Noetherian. -/
theorem isLocallyNoetherian_of_affine_cover {ι} {S : ι → X.affineOpens}
    (hS : (⨆ i, S i : X.Opens) = ⊤)
    (hS' : ∀ i, IsNoetherianRing Γ(X, S i)) : IsLocallyNoetherian X := by
  /-
    X : AlgebraicGeometry.Scheme
    ι : Sort u_1
    S : ι → ↑X.affineOpens
    hS : Eq (iSup fun i => ↑(S i)) Top.top
    hS' : ∀ (i : ι), IsNoetherianRing ↑(X.presheaf.obj { unop := ↑(S i) })
    ⊢ AlgebraicGeometry.IsLocallyNoetherian X
  -/
  refine ⟨fun U => ?_⟩
  induction U using of_affine_open_cover S hS with
  | basicOpen U f hN =>
    have := U.prop.isLocalization_basicOpen f
    exact IsLocalization.isNoetherianRing (.powers f) Γ(X, X.basicOpen f) hN
  | openCover U s _ hN =>
    apply isNoetherianRing_of_away s ‹_›
    intro ⟨f, hf⟩
    have : IsNoetherianRing Γ(X, X.basicOpen f) := hN ⟨f, hf⟩
    have := U.prop.isLocalization_basicOpen f
    have hEq := IsLocalization.algEquiv (.powers f) (Localization.Away f) Γ(X, X.basicOpen f)
    exact isNoetherianRing_of_ringEquiv Γ(X, X.basicOpen f) hEq.symm.toRingEquiv
  | hU => exact hS' _


/-- A scheme is locally Noetherian if and only if it is covered by affine opens whose sections
are noetherian rings.

See [Har77], Proposition II.3.2. -/
theorem isLocallyNoetherian_iff_of_iSup_eq_top {ι} {S : ι → X.affineOpens}
    (hS : (⨆ i, S i : X.Opens) = ⊤) :
    IsLocallyNoetherian X ↔ ∀ i, IsNoetherianRing Γ(X, S i) :=
  ⟨fun _ i => IsLocallyNoetherian.component_noetherian (S i),
   isLocallyNoetherian_of_affine_cover hS⟩


open CategoryTheory in
/-- A version of `isLocallyNoetherian_iff_of_iSup_eq_top` using `Scheme.OpenCover`. -/
theorem isLocallyNoetherian_iff_of_affine_openCover (𝒰 : Scheme.OpenCover.{v, u} X)
    [∀ i, IsAffine (𝒰.obj i)] :
    IsLocallyNoetherian X ↔ ∀ (i : 𝒰.J), IsNoetherianRing Γ(𝒰.obj i, ⊤) := by
  /-
    X : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    inst✝ : ∀ (i : 𝒰.J), AlgebraicGeometry.IsAffine (𝒰.obj i)
    ⊢ Iff (AlgebraicGeometry.IsLocallyNoetherian X) (∀ (i : 𝒰.J), IsNoetherianRing …
  -/
  constructor
    /-
      case mp
      X : AlgebraicGeometry.Scheme
      𝒰 : X.OpenCover
      inst✝ : ∀ (i : 𝒰.J), AlgebraicGeometry.IsAffine (𝒰.obj i)
      ⊢ AlgebraicGeometry.IsLocallyNoetherian X → ∀ (i : 𝒰.J), IsNoetherianRing ↑((𝒰 …
    -/
  · intro h i
    /-
      case mp
      X : AlgebraicGeometry.Scheme
      𝒰 : X.OpenCover
      inst✝ : ∀ (i : 𝒰.J), AlgebraicGeometry.IsAffine (𝒰.obj i)
      h : AlgebraicGeometry.IsLocallyNoetherian X
      i : 𝒰.J
      ⊢ IsNoetherianRing ↑((𝒰.obj i).presheaf.obj { unop := Top.top })
    -/
    let U := Scheme.Hom.opensRange (𝒰.map i)
    /-
      case mp
      X : AlgebraicGeometry.Scheme
      𝒰 : X.OpenCover
      inst✝ : ∀ (i : 𝒰.J), AlgebraicGeometry.IsAffine (𝒰.obj i)
      h : AlgebraicGeometry.IsLocallyNoetherian X
      i : 𝒰.J
      U : X.Opens := AlgebraicGeometry.Scheme.Hom.opensRange (𝒰.map i)
      ⊢ IsNoetherianRing ↑((𝒰.obj i).presheaf.obj { unop := Top.top })
    -/
    have := h.component_noetherian ⟨U, isAffineOpen_opensRange _⟩
    /-
      case mp
      X : AlgebraicGeometry.Scheme
      𝒰 : X.OpenCover
      inst✝ : ∀ (i : 𝒰.J), AlgebraicGeometry.IsAffine (𝒰.obj i)
      h : AlgebraicGeometry.IsLocallyNoetherian X
      i : 𝒰.J
      U : X.Opens := AlgebraicGeometry.Scheme.Hom.opensRange (𝒰.map i)
      this : IsNoetherianRing ↑(X.presheaf.obj { unop := ↑⟨U, ⋯⟩ })
      ⊢ IsNoetherianRing ↑((𝒰.obj i).presheaf.obj { unop := Top.top })
    -/
    apply isNoetherianRing_of_ringEquiv (R := Γ(X, U))
    /-
      case mp.f
      X : AlgebraicGeometry.Scheme
      𝒰 : X.OpenCover
      inst✝ : ∀ (i : 𝒰.J), AlgebraicGeometry.IsAffine (𝒰.obj i)
      h : AlgebraicGeometry.IsLocallyNoetherian X
      i : 𝒰.J
      U : X.Opens := AlgebraicGeometry.Scheme.Hom.opensRange (𝒰.map i)
      this : IsNoetherianRing ↑(X.presheaf.obj { unop := ↑⟨U, ⋯⟩ })
      ⊢ RingEquiv ↑(X.presheaf.obj { unop := U }) ↑((𝒰.obj i).presheaf.obj { unop := …
    -/
    apply CategoryTheory.Iso.commRingCatIsoToRingEquiv
    /-
      case mp.f.e
      X : AlgebraicGeometry.Scheme
      𝒰 : X.OpenCover
      inst✝ : ∀ (i : 𝒰.J), AlgebraicGeometry.IsAffine (𝒰.obj i)
      h : AlgebraicGeometry.IsLocallyNoetherian X
      i : 𝒰.J
      U : X.Opens := AlgebraicGeometry.Scheme.Hom.opensRange (𝒰.map i)
      this : IsNoetherianRing ↑(X.presheaf.obj { unop := ↑⟨U, ⋯⟩ })
      ⊢ CategoryTheory.Iso (X.presheaf.obj { unop := U }) ((𝒰.obj i).presheaf.obj {  …
    -/
    exact (IsOpenImmersion.ΓIsoTop (𝒰.map i)).symm
    /-
      🎉 no goals
    -/
    /-
      case mpr
      X : AlgebraicGeometry.Scheme
      𝒰 : X.OpenCover
      inst✝ : ∀ (i : 𝒰.J), AlgebraicGeometry.IsAffine (𝒰.obj i)
      ⊢ (∀ (i : 𝒰.J), IsNoetherianRing ↑((𝒰.obj i).presheaf.obj { unop := Top.top }) …
    -/
  · intro hCNoeth
    /-
      case mpr
      X : AlgebraicGeometry.Scheme
      𝒰 : X.OpenCover
      inst✝ : ∀ (i : 𝒰.J), AlgebraicGeometry.IsAffine (𝒰.obj i)
      hCNoeth : ∀ (i : 𝒰.J), IsNoetherianRing ↑((𝒰.obj i).presheaf.obj { unop := Top …
      ⊢ AlgebraicGeometry.IsLocallyNoetherian X
    -/
    let fS i : X.affineOpens := ⟨Scheme.Hom.opensRange (𝒰.map i), isAffineOpen_opensRange _⟩
    /-
      case mpr
      X : AlgebraicGeometry.Scheme
      𝒰 : X.OpenCover
      inst✝ : ∀ (i : 𝒰.J), AlgebraicGeometry.IsAffine (𝒰.obj i)
      hCNoeth : ∀ (i : 𝒰.J), IsNoetherianRing ↑((𝒰.obj i).presheaf.obj { unop := Top …
      fS : 𝒰.J → ↑X.affineOpens := fun i => ⟨AlgebraicGeometry.Scheme.Hom.opensRange …
      ⊢ AlgebraicGeometry.IsLocallyNoetherian X
    -/
    apply isLocallyNoetherian_of_affine_cover (S := fS)
      /-
        case mpr.hS
        X : AlgebraicGeometry.Scheme
        𝒰 : X.OpenCover
        inst✝ : ∀ (i : 𝒰.J), AlgebraicGeometry.IsAffine (𝒰.obj i)
        hCNoeth : ∀ (i : 𝒰.J), IsNoetherianRing ↑((𝒰.obj i).presheaf.obj { unop := Top …
        fS : 𝒰.J → ↑X.affineOpens := fun i => ⟨AlgebraicGeometry.Scheme.Hom.opensRange …
        ⊢ Eq (iSup fun i => ↑(fS i)) Top.top
      -/
    · rw [← Scheme.OpenCover.iSup_opensRange 𝒰]
      /-
        🎉 no goals
      -/
    /-
      case mpr.hS'
      X : AlgebraicGeometry.Scheme
      𝒰 : X.OpenCover
      inst✝ : ∀ (i : 𝒰.J), AlgebraicGeometry.IsAffine (𝒰.obj i)
      hCNoeth : ∀ (i : 𝒰.J), IsNoetherianRing ↑((𝒰.obj i).presheaf.obj { unop := Top …
      fS : 𝒰.J → ↑X.affineOpens := fun i => ⟨AlgebraicGeometry.Scheme.Hom.opensRange …
      ⊢ ∀ (i : 𝒰.J), IsNoetherianRing ↑(X.presheaf.obj { unop := ↑(fS i) })
    -/
    intro i
    /-
      case mpr.hS'
      X : AlgebraicGeometry.Scheme
      𝒰 : X.OpenCover
      inst✝ : ∀ (i : 𝒰.J), AlgebraicGeometry.IsAffine (𝒰.obj i)
      hCNoeth : ∀ (i : 𝒰.J), IsNoetherianRing ↑((𝒰.obj i).presheaf.obj { unop := Top …
      fS : 𝒰.J → ↑X.affineOpens := fun i => ⟨AlgebraicGeometry.Scheme.Hom.opensRange …
      i : 𝒰.J
      ⊢ IsNoetherianRing ↑(X.presheaf.obj { unop := ↑(fS i) })
    -/
    apply isNoetherianRing_of_ringEquiv (R := Γ(𝒰.obj i, ⊤))
    /-
      case mpr.hS'.f
      X : AlgebraicGeometry.Scheme
      𝒰 : X.OpenCover
      inst✝ : ∀ (i : 𝒰.J), AlgebraicGeometry.IsAffine (𝒰.obj i)
      hCNoeth : ∀ (i : 𝒰.J), IsNoetherianRing ↑((𝒰.obj i).presheaf.obj { unop := Top …
      fS : 𝒰.J → ↑X.affineOpens := fun i => ⟨AlgebraicGeometry.Scheme.Hom.opensRange …
      i : 𝒰.J
      ⊢ RingEquiv ↑((𝒰.obj i).presheaf.obj { unop := Top.top }) ↑(X.presheaf.obj { u …
    -/
    apply CategoryTheory.Iso.commRingCatIsoToRingEquiv
    /-
      case mpr.hS'.f.e
      X : AlgebraicGeometry.Scheme
      𝒰 : X.OpenCover
      inst✝ : ∀ (i : 𝒰.J), AlgebraicGeometry.IsAffine (𝒰.obj i)
      hCNoeth : ∀ (i : 𝒰.J), IsNoetherianRing ↑((𝒰.obj i).presheaf.obj { unop := Top …
      fS : 𝒰.J → ↑X.affineOpens := fun i => ⟨AlgebraicGeometry.Scheme.Hom.opensRange …
      i : 𝒰.J
      ⊢ CategoryTheory.Iso ((𝒰.obj i).presheaf.obj { unop := Top.top }) (X.presheaf. …
    -/
    exact IsOpenImmersion.ΓIsoTop (𝒰.map i)
    /-
      🎉 no goals
    -/


lemma isLocallyNoetherian_of_isOpenImmersion {Y : Scheme} (f : X ⟶ Y) [IsOpenImmersion f]
    [IsLocallyNoetherian Y] : IsLocallyNoetherian X := by
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    inst✝¹ : AlgebraicGeometry.IsOpenImmersion f
    inst✝ : AlgebraicGeometry.IsLocallyNoetherian Y
    ⊢ AlgebraicGeometry.IsLocallyNoetherian X
  -/
  refine ⟨fun U => ?_⟩
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    inst✝¹ : AlgebraicGeometry.IsOpenImmersion f
    inst✝ : AlgebraicGeometry.IsLocallyNoetherian Y
    U : ↑X.affineOpens
    ⊢ IsNoetherianRing ↑(X.presheaf.obj { unop := ↑U })
  -/
  let V : Y.affineOpens := ⟨f ''ᵁ U, IsAffineOpen.image_of_isOpenImmersion U.prop _⟩
  suffices Γ(X, U) ≅ Γ(Y, V) by
    convert isNoetherianRing_of_ringEquiv (R := Γ(Y, V)) _
    · apply CategoryTheory.Iso.commRingCatIsoToRingEquiv
      exact this.symm
    · exact IsLocallyNoetherian.component_noetherian V
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    inst✝¹ : AlgebraicGeometry.IsOpenImmersion f
    inst✝ : AlgebraicGeometry.IsLocallyNoetherian Y
    U : ↑X.affineOpens
    V : ↑Y.affineOpens := ⟨(AlgebraicGeometry.Scheme.Hom.opensFunctor f).obj ↑U, ⋯⟩
    ⊢ CategoryTheory.Iso (X.presheaf.obj { unop := ↑U }) (Y.presheaf.obj { unop := …
  -/
  rw [← Scheme.Hom.preimage_image_eq f U]
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    inst✝¹ : AlgebraicGeometry.IsOpenImmersion f
    inst✝ : AlgebraicGeometry.IsLocallyNoetherian Y
    U : ↑X.affineOpens
    V : ↑Y.affineOpens := ⟨(AlgebraicGeometry.Scheme.Hom.opensFunctor f).obj ↑U, ⋯⟩
    ⊢ CategoryTheory.Iso (X.presheaf.obj { unop := (TopologicalSpace.Opens.map f.b …
  -/
  trans
    /-
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      inst✝¹ : AlgebraicGeometry.IsOpenImmersion f
      inst✝ : AlgebraicGeometry.IsLocallyNoetherian Y
      U : ↑X.affineOpens
      V : ↑Y.affineOpens := ⟨(AlgebraicGeometry.Scheme.Hom.opensFunctor f).obj ↑U, ⋯⟩
      ⊢ CategoryTheory.Iso (X.presheaf.obj { unop := (TopologicalSpace.Opens.map f.b …
    -/
  · apply IsOpenImmersion.ΓIso
    /-
      🎉 no goals
    -/
  · suffices Scheme.Hom.opensRange f ⊓ V = V by
      rw [this]
    /-
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      inst✝¹ : AlgebraicGeometry.IsOpenImmersion f
      inst✝ : AlgebraicGeometry.IsLocallyNoetherian Y
      U : ↑X.affineOpens
      V : ↑Y.affineOpens := ⟨(AlgebraicGeometry.Scheme.Hom.opensFunctor f).obj ↑U, ⋯⟩
      ⊢ Eq (Min.min (AlgebraicGeometry.Scheme.Hom.opensRange f) ↑V) ↑V
    -/
    rw [← Opens.coe_inj]
    rw [Opens.coe_inf, Scheme.Hom.coe_opensRange, IsOpenMap.coe_functor_obj,
      Set.inter_eq_right, Set.image_subset_iff, Set.preimage_range]
    /-
      X Y : AlgebraicGeometry.Scheme
      f : Quiver.Hom X Y
      inst✝¹ : AlgebraicGeometry.IsOpenImmersion f
      inst✝ : AlgebraicGeometry.IsLocallyNoetherian Y
      U : ↑X.affineOpens
      V : ↑Y.affineOpens := ⟨(AlgebraicGeometry.Scheme.Hom.opensFunctor f).obj ↑U, ⋯⟩
      ⊢ HasSubset.Subset (↑↑U) Set.univ
    -/
    exact Set.subset_univ _
    /-
      🎉 no goals
    -/


/-- If `𝒰` is an open cover of a scheme `X`, then `X` is locally noetherian if and only if
`𝒰.obj i` are all locally noetherian. -/
theorem isLocallyNoetherian_iff_openCover (𝒰 : Scheme.OpenCover X) :
    IsLocallyNoetherian X ↔ ∀ (i : 𝒰.J), IsLocallyNoetherian (𝒰.obj i) := by
  /-
    X : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    ⊢ Iff (AlgebraicGeometry.IsLocallyNoetherian X) (∀ (i : 𝒰.J), AlgebraicGeometr …
  -/
  constructor
    /-
      case mp
      X : AlgebraicGeometry.Scheme
      𝒰 : X.OpenCover
      ⊢ AlgebraicGeometry.IsLocallyNoetherian X → ∀ (i : 𝒰.J), AlgebraicGeometry.IsL …
    -/
  · intro h i
    /-
      case mp
      X : AlgebraicGeometry.Scheme
      𝒰 : X.OpenCover
      h : AlgebraicGeometry.IsLocallyNoetherian X
      i : 𝒰.J
      ⊢ AlgebraicGeometry.IsLocallyNoetherian (𝒰.obj i)
    -/
    exact isLocallyNoetherian_of_isOpenImmersion (𝒰.map i)
    /-
      🎉 no goals
    -/
    /-
      case mpr
      X : AlgebraicGeometry.Scheme
      𝒰 : X.OpenCover
      ⊢ (∀ (i : 𝒰.J), AlgebraicGeometry.IsLocallyNoetherian (𝒰.obj i)) → AlgebraicGe …
    -/
  · rw [isLocallyNoetherian_iff_of_affine_openCover (𝒰 := 𝒰.affineRefinement.openCover)]
    /-
      case mpr
      X : AlgebraicGeometry.Scheme
      𝒰 : X.OpenCover
      ⊢ (∀ (i : 𝒰.J), AlgebraicGeometry.IsLocallyNoetherian (𝒰.obj i)) → ∀ (i : 𝒰.af …
    -/
    intro h i
    exact @isNoetherianRing_of_ringEquiv _ _ _ _
      (IsOpenImmersion.ΓIsoTop (Scheme.Cover.map _ i.2)).symm.commRingCatIsoToRingEquiv
      (IsLocallyNoetherian.component_noetherian ⟨_, isAffineOpen_opensRange _⟩)


/-- If `R` is a noetherian ring, `Spec R` is a noetherian topological space. -/
instance {R : CommRingCat} [IsNoetherianRing R] :
    NoetherianSpace (Spec R) := by
  /-
    X : AlgebraicGeometry.Scheme
    R : CommRingCat
    inst✝ : IsNoetherianRing ↑R
    ⊢ TopologicalSpace.NoetherianSpace ↑↑(AlgebraicGeometry.Spec R).toPresheafedSp …
  -/
  convert PrimeSpectrum.instNoetherianSpace (R := R)
  /-
    🎉 no goals
  -/


lemma noetherianSpace_of_isAffine [IsAffine X] [IsNoetherianRing Γ(X, ⊤)] :
    NoetherianSpace X :=
  (noetherianSpace_iff_of_homeomorph X.isoSpec.inv.homeomorph).mp inferInstance


lemma noetherianSpace_of_isAffineOpen (U : X.Opens) (hU : IsAffineOpen U)
    [IsNoetherianRing Γ(X, U)] :
    NoetherianSpace U := by
  have : IsNoetherianRing Γ(U, ⊤) := isNoetherianRing_of_ringEquiv _
    (Scheme.restrictFunctorΓ.app (op U)).symm.commRingCatIsoToRingEquiv
  /-
    X : AlgebraicGeometry.Scheme
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    inst✝ : IsNoetherianRing ↑(X.presheaf.obj { unop := U })
    this : IsNoetherianRing ↑((↑U).presheaf.obj { unop := Top.top })
    ⊢ TopologicalSpace.NoetherianSpace ↑↑(↑U).toPresheafedSpace
  -/
  exact @noetherianSpace_of_isAffine _ hU _
  /-
    🎉 no goals
  -/


/-- Any open immersion `Z ⟶ X` with `X` locally Noetherian is quasi-compact.

[Stacks: Lemma 01OX](https://stacks.math.columbia.edu/tag/01OX) -/
instance (priority := 100) {Z : Scheme} [IsLocallyNoetherian X]
    {f : Z ⟶ X} [IsOpenImmersion f] : QuasiCompact f := by
  /-
    X Z : AlgebraicGeometry.Scheme
    inst✝¹ : AlgebraicGeometry.IsLocallyNoetherian X
    f : Quiver.Hom Z X
    inst✝ : AlgebraicGeometry.IsOpenImmersion f
    ⊢ AlgebraicGeometry.QuasiCompact f
  -/
  apply (quasiCompact_iff_forall_affine f).mpr
  /-
    X Z : AlgebraicGeometry.Scheme
    inst✝¹ : AlgebraicGeometry.IsLocallyNoetherian X
    f : Quiver.Hom Z X
    inst✝ : AlgebraicGeometry.IsOpenImmersion f
    ⊢ ∀ (U : X.Opens), AlgebraicGeometry.IsAffineOpen U → IsCompact ↑((Topological …
  -/
  intro U hU
  /-
    X Z : AlgebraicGeometry.Scheme
    inst✝¹ : AlgebraicGeometry.IsLocallyNoetherian X
    f : Quiver.Hom Z X
    inst✝ : AlgebraicGeometry.IsOpenImmersion f
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    ⊢ IsCompact ↑((TopologicalSpace.Opens.map f.base).obj U)
  -/
  rw [Opens.map_coe, ← Set.preimage_inter_range]
  /-
    X Z : AlgebraicGeometry.Scheme
    inst✝¹ : AlgebraicGeometry.IsLocallyNoetherian X
    f : Quiver.Hom Z X
    inst✝ : AlgebraicGeometry.IsOpenImmersion f
    U : X.Opens
    hU : AlgebraicGeometry.IsAffineOpen U
    ⊢ IsCompact (Set.preimage (⇑f.base) (Inter.inter (↑U) (Set.range ⇑f.base)))
  -/
  apply f.isOpenEmbedding.isInducing.isCompact_preimage'
    /-
      case hK
      X Z : AlgebraicGeometry.Scheme
      inst✝¹ : AlgebraicGeometry.IsLocallyNoetherian X
      f : Quiver.Hom Z X
      inst✝ : AlgebraicGeometry.IsOpenImmersion f
      U : X.Opens
      hU : AlgebraicGeometry.IsAffineOpen U
      ⊢ IsCompact (Inter.inter (↑U) (Set.range ⇑f.base))
    -/
  · apply (noetherianSpace_set_iff _).mp
      /-
        case hK.a
        X Z : AlgebraicGeometry.Scheme
        inst✝¹ : AlgebraicGeometry.IsLocallyNoetherian X
        f : Quiver.Hom Z X
        inst✝ : AlgebraicGeometry.IsOpenImmersion f
        U : X.Opens
        hU : AlgebraicGeometry.IsAffineOpen U
        ⊢ TopologicalSpace.NoetherianSpace ↑?m.41268
      -/
    · convert noetherianSpace_of_isAffineOpen U hU
      /-
        case hK.a
        X Z : AlgebraicGeometry.Scheme
        inst✝¹ : AlgebraicGeometry.IsLocallyNoetherian X
        f : Quiver.Hom Z X
        inst✝ : AlgebraicGeometry.IsOpenImmersion f
        U : X.Opens
        hU : AlgebraicGeometry.IsAffineOpen U
        ⊢ IsNoetherianRing ↑(X.presheaf.obj { unop := U })
      -/
      apply IsLocallyNoetherian.component_noetherian ⟨U, hU⟩
      /-
        🎉 no goals
      -/
      /-
        case hK.a
        X Z : AlgebraicGeometry.Scheme
        inst✝¹ : AlgebraicGeometry.IsLocallyNoetherian X
        f : Quiver.Hom Z X
        inst✝ : AlgebraicGeometry.IsOpenImmersion f
        U : X.Opens
        hU : AlgebraicGeometry.IsAffineOpen U
        ⊢ HasSubset.Subset (Inter.inter (↑U) (Set.range ⇑f.base)) ↑U
      -/
    · exact Set.inter_subset_left
      /-
        🎉 no goals
      -/
    /-
      case Kf
      X Z : AlgebraicGeometry.Scheme
      inst✝¹ : AlgebraicGeometry.IsLocallyNoetherian X
      f : Quiver.Hom Z X
      inst✝ : AlgebraicGeometry.IsOpenImmersion f
      U : X.Opens
      hU : AlgebraicGeometry.IsAffineOpen U
      ⊢ HasSubset.Subset (Inter.inter (↑U) (Set.range ⇑f.base)) (Set.range ⇑f.base)
    -/
  · exact Set.inter_subset_right
    /-
      🎉 no goals
    -/


/-- A locally Noetherian scheme is quasi-separated.

[Stacks: Lemma 01OY](https://stacks.math.columbia.edu/tag/01OY) -/
instance (priority := 100) IsLocallyNoetherian.quasiSeparatedSpace [IsLocallyNoetherian X] :
    QuasiSeparatedSpace X := by
  /-
    X : AlgebraicGeometry.Scheme
    inst✝ : AlgebraicGeometry.IsLocallyNoetherian X
    ⊢ QuasiSeparatedSpace ↑↑X.toPresheafedSpace
  -/
  apply (quasiSeparatedSpace_iff_affine X).mpr
  /-
    X : AlgebraicGeometry.Scheme
    inst✝ : AlgebraicGeometry.IsLocallyNoetherian X
    ⊢ ∀ (U V : ↑X.affineOpens), IsCompact (Inter.inter ↑↑U ↑↑V)
  -/
  intro U V
  /-
    X : AlgebraicGeometry.Scheme
    inst✝ : AlgebraicGeometry.IsLocallyNoetherian X
    U V : ↑X.affineOpens
    ⊢ IsCompact (Inter.inter ↑↑U ↑↑V)
  -/
  have hInd := U.2.fromSpec.isOpenEmbedding.isInducing
  /-
    X : AlgebraicGeometry.Scheme
    inst✝ : AlgebraicGeometry.IsLocallyNoetherian X
    U V : ↑X.affineOpens
    hInd : Topology.IsInducing ⇑(AlgebraicGeometry.IsAffineOpen.fromSpec ⋯).base
    ⊢ IsCompact (Inter.inter ↑↑U ↑↑V)
  -/
  apply (hInd.isCompact_preimage_iff ?_).mp
    /-
      X : AlgebraicGeometry.Scheme
      inst✝ : AlgebraicGeometry.IsLocallyNoetherian X
      U V : ↑X.affineOpens
      hInd : Topology.IsInducing ⇑(AlgebraicGeometry.IsAffineOpen.fromSpec ⋯).base
      ⊢ IsCompact (Set.preimage (⇑(AlgebraicGeometry.IsAffineOpen.fromSpec ⋯).base)  …
    -/
  · rw [← Set.preimage_inter_range, IsAffineOpen.range_fromSpec, Set.inter_comm]
    /-
      X : AlgebraicGeometry.Scheme
      inst✝ : AlgebraicGeometry.IsLocallyNoetherian X
      U V : ↑X.affineOpens
      hInd : Topology.IsInducing ⇑(AlgebraicGeometry.IsAffineOpen.fromSpec ⋯).base
      ⊢ IsCompact (Set.preimage (⇑(AlgebraicGeometry.IsAffineOpen.fromSpec ⋯).base)  …
    -/
    apply hInd.isCompact_preimage'
      /-
        case hK
        X : AlgebraicGeometry.Scheme
        inst✝ : AlgebraicGeometry.IsLocallyNoetherian X
        U V : ↑X.affineOpens
        hInd : Topology.IsInducing ⇑(AlgebraicGeometry.IsAffineOpen.fromSpec ⋯).base
        ⊢ IsCompact (Inter.inter (↑↑U) (Inter.inter ↑↑U ↑↑V))
      -/
    · apply (noetherianSpace_set_iff _).mp
        /-
          case hK.a
          X : AlgebraicGeometry.Scheme
          inst✝ : AlgebraicGeometry.IsLocallyNoetherian X
          U V : ↑X.affineOpens
          hInd : Topology.IsInducing ⇑(AlgebraicGeometry.IsAffineOpen.fromSpec ⋯).base
          ⊢ TopologicalSpace.NoetherianSpace ↑?m.44345
        -/
      · convert noetherianSpace_of_isAffineOpen U.1 U.2
        /-
          case hK.a
          X : AlgebraicGeometry.Scheme
          inst✝ : AlgebraicGeometry.IsLocallyNoetherian X
          U V : ↑X.affineOpens
          hInd : Topology.IsInducing ⇑(AlgebraicGeometry.IsAffineOpen.fromSpec ⋯).base
          ⊢ IsNoetherianRing ↑(X.presheaf.obj { unop := ↑U })
        -/
        apply IsLocallyNoetherian.component_noetherian
        /-
          🎉 no goals
        -/
        /-
          case hK.a
          X : AlgebraicGeometry.Scheme
          inst✝ : AlgebraicGeometry.IsLocallyNoetherian X
          U V : ↑X.affineOpens
          hInd : Topology.IsInducing ⇑(AlgebraicGeometry.IsAffineOpen.fromSpec ⋯).base
          ⊢ HasSubset.Subset (Inter.inter (↑↑U) (Inter.inter ↑↑U ↑↑V)) ↑↑U
        -/
      · exact Set.inter_subset_left
        /-
          🎉 no goals
        -/
      /-
        case Kf
        X : AlgebraicGeometry.Scheme
        inst✝ : AlgebraicGeometry.IsLocallyNoetherian X
        U V : ↑X.affineOpens
        hInd : Topology.IsInducing ⇑(AlgebraicGeometry.IsAffineOpen.fromSpec ⋯).base
        ⊢ HasSubset.Subset (Inter.inter (↑↑U) (Inter.inter ↑↑U ↑↑V)) (Set.range ⇑(Alge …
      -/
    · rw [IsAffineOpen.range_fromSpec]
      /-
        case Kf
        X : AlgebraicGeometry.Scheme
        inst✝ : AlgebraicGeometry.IsLocallyNoetherian X
        U V : ↑X.affineOpens
        hInd : Topology.IsInducing ⇑(AlgebraicGeometry.IsAffineOpen.fromSpec ⋯).base
        ⊢ HasSubset.Subset (Inter.inter (↑↑U) (Inter.inter ↑↑U ↑↑V)) ↑↑U
      -/
      exact Set.inter_subset_left
      /-
        🎉 no goals
      -/
    /-
      X : AlgebraicGeometry.Scheme
      inst✝ : AlgebraicGeometry.IsLocallyNoetherian X
      U V : ↑X.affineOpens
      hInd : Topology.IsInducing ⇑(AlgebraicGeometry.IsAffineOpen.fromSpec ⋯).base
      ⊢ HasSubset.Subset (Inter.inter ↑↑U ↑↑V) (Set.range ⇑(AlgebraicGeometry.IsAffi …
    -/
  · rw [IsAffineOpen.range_fromSpec]
    /-
      X : AlgebraicGeometry.Scheme
      inst✝ : AlgebraicGeometry.IsLocallyNoetherian X
      U V : ↑X.affineOpens
      hInd : Topology.IsInducing ⇑(AlgebraicGeometry.IsAffineOpen.fromSpec ⋯).base
      ⊢ HasSubset.Subset (Inter.inter ↑↑U ↑↑V) ↑↑U
    -/
    exact Set.inter_subset_left
    /-
      🎉 no goals
    -/


/-- A scheme `X` is Noetherian if it is locally Noetherian and compact. -/
@[mk_iff]
class IsNoetherian (X : Scheme) extends IsLocallyNoetherian X, CompactSpace X : Prop


/-- A scheme is Noetherian if and only if it is covered by finitely many affine opens whose
sections are noetherian rings. -/
theorem isNoetherian_iff_of_finite_iSup_eq_top {ι} [Finite ι] {S : ι → X.affineOpens}
    (hS : (⨆ i, S i : X.Opens) = ⊤) :
    IsNoetherian X ↔ ∀ i, IsNoetherianRing Γ(X, S i) := by
  /-
    X : AlgebraicGeometry.Scheme
    ι : Sort u_1
    inst✝ : Finite ι
    S : ι → ↑X.affineOpens
    hS : Eq (iSup fun i => ↑(S i)) Top.top
    ⊢ Iff (AlgebraicGeometry.IsNoetherian X) (∀ (i : ι), IsNoetherianRing ↑(X.pres …
  -/
  constructor
    /-
      case mp
      X : AlgebraicGeometry.Scheme
      ι : Sort u_1
      inst✝ : Finite ι
      S : ι → ↑X.affineOpens
      hS : Eq (iSup fun i => ↑(S i)) Top.top
      ⊢ AlgebraicGeometry.IsNoetherian X → ∀ (i : ι), IsNoetherianRing ↑(X.presheaf. …
    -/
  · intro h i
    /-
      case mp
      X : AlgebraicGeometry.Scheme
      ι : Sort u_1
      inst✝ : Finite ι
      S : ι → ↑X.affineOpens
      hS : Eq (iSup fun i => ↑(S i)) Top.top
      h : AlgebraicGeometry.IsNoetherian X
      i : ι
      ⊢ IsNoetherianRing ↑(X.presheaf.obj { unop := ↑(S i) })
    -/
    apply (isLocallyNoetherian_iff_of_iSup_eq_top hS).mp
    /-
      case mp.a
      X : AlgebraicGeometry.Scheme
      ι : Sort u_1
      inst✝ : Finite ι
      S : ι → ↑X.affineOpens
      hS : Eq (iSup fun i => ↑(S i)) Top.top
      h : AlgebraicGeometry.IsNoetherian X
      i : ι
      ⊢ AlgebraicGeometry.IsLocallyNoetherian X
    -/
    exact h.toIsLocallyNoetherian
    /-
      🎉 no goals
    -/
    /-
      case mpr
      X : AlgebraicGeometry.Scheme
      ι : Sort u_1
      inst✝ : Finite ι
      S : ι → ↑X.affineOpens
      hS : Eq (iSup fun i => ↑(S i)) Top.top
      ⊢ (∀ (i : ι), IsNoetherianRing ↑(X.presheaf.obj { unop := ↑(S i) })) → Algebra …
    -/
  · intro h
    /-
      case mpr
      X : AlgebraicGeometry.Scheme
      ι : Sort u_1
      inst✝ : Finite ι
      S : ι → ↑X.affineOpens
      hS : Eq (iSup fun i => ↑(S i)) Top.top
      h : ∀ (i : ι), IsNoetherianRing ↑(X.presheaf.obj { unop := ↑(S i) })
      ⊢ AlgebraicGeometry.IsNoetherian X
    -/
    convert IsNoetherian.mk
      /-
        case mpr.convert_2
        X : AlgebraicGeometry.Scheme
        ι : Sort u_1
        inst✝ : Finite ι
        S : ι → ↑X.affineOpens
        hS : Eq (iSup fun i => ↑(S i)) Top.top
        h : ∀ (i : ι), IsNoetherianRing ↑(X.presheaf.obj { unop := ↑(S i) })
        ⊢ AlgebraicGeometry.IsLocallyNoetherian X
      -/
    · exact isLocallyNoetherian_of_affine_cover hS h
      /-
        🎉 no goals
      -/
      /-
        case mpr.convert_3
        X : AlgebraicGeometry.Scheme
        ι : Sort u_1
        inst✝ : Finite ι
        S : ι → ↑X.affineOpens
        hS : Eq (iSup fun i => ↑(S i)) Top.top
        h : ∀ (i : ι), IsNoetherianRing ↑(X.presheaf.obj { unop := ↑(S i) })
        ⊢ CompactSpace ↑↑X.toPresheafedSpace
      -/
    · constructor
      /-
        case mpr.convert_3.isCompact_univ
        X : AlgebraicGeometry.Scheme
        ι : Sort u_1
        inst✝ : Finite ι
        S : ι → ↑X.affineOpens
        hS : Eq (iSup fun i => ↑(S i)) Top.top
        h : ∀ (i : ι), IsNoetherianRing ↑(X.presheaf.obj { unop := ↑(S i) })
        ⊢ IsCompact Set.univ
      -/
      rw [← Opens.coe_top, ← hS, Opens.iSup_mk]
      /-
        case mpr.convert_3.isCompact_univ
        X : AlgebraicGeometry.Scheme
        ι : Sort u_1
        inst✝ : Finite ι
        S : ι → ↑X.affineOpens
        hS : Eq (iSup fun i => ↑(S i)) Top.top
        h : ∀ (i : ι), IsNoetherianRing ↑(X.presheaf.obj { unop := ↑(S i) })
        ⊢ IsCompact ↑{ carrier := Set.iUnion fun i => (S i).1.carrier, is_open' := ⋯ }
      -/
      apply isCompact_iUnion
      /-
        case mpr.convert_3.isCompact_univ.h
        X : AlgebraicGeometry.Scheme
        ι : Sort u_1
        inst✝ : Finite ι
        S : ι → ↑X.affineOpens
        hS : Eq (iSup fun i => ↑(S i)) Top.top
        h : ∀ (i : ι), IsNoetherianRing ↑(X.presheaf.obj { unop := ↑(S i) })
        ⊢ ∀ (i : ι), IsCompact (S i).1.carrier
      -/
      intro i
      /-
        case mpr.convert_3.isCompact_univ.h
        X : AlgebraicGeometry.Scheme
        ι : Sort u_1
        inst✝ : Finite ι
        S : ι → ↑X.affineOpens
        hS : Eq (iSup fun i => ↑(S i)) Top.top
        h : ∀ (i : ι), IsNoetherianRing ↑(X.presheaf.obj { unop := ↑(S i) })
        i : ι
        ⊢ IsCompact (S i).1.carrier
      -/
      apply isCompact_iff_isCompact_univ.mpr
      /-
        case mpr.convert_3.isCompact_univ.h
        X : AlgebraicGeometry.Scheme
        ι : Sort u_1
        inst✝ : Finite ι
        S : ι → ↑X.affineOpens
        hS : Eq (iSup fun i => ↑(S i)) Top.top
        h : ∀ (i : ι), IsNoetherianRing ↑(X.presheaf.obj { unop := ↑(S i) })
        i : ι
        ⊢ IsCompact Set.univ
      -/
      convert CompactSpace.isCompact_univ
      have : NoetherianSpace (S i) := by
        apply noetherianSpace_of_isAffineOpen (S i).1 (S i).2
      /-
        case mpr.convert_3.isCompact_univ.h.convert_3
        X : AlgebraicGeometry.Scheme
        ι : Sort u_1
        inst✝ : Finite ι
        S : ι → ↑X.affineOpens
        hS : Eq (iSup fun i => ↑(S i)) Top.top
        h : ∀ (i : ι), IsNoetherianRing ↑(X.presheaf.obj { unop := ↑(S i) })
        i : ι
        this : TopologicalSpace.NoetherianSpace ↑↑(↑↑(S i)).toPresheafedSpace
        ⊢ CompactSpace ↑(S i).1.carrier
      -/
      apply NoetherianSpace.compactSpace (S i)
      /-
        🎉 no goals
      -/


/-- A version of `isNoetherian_iff_of_finite_iSup_eq_top` using `Scheme.OpenCover`. -/
theorem isNoetherian_iff_of_finite_affine_openCover {𝒰 : Scheme.OpenCover.{v, u} X}
    [Finite 𝒰.J] [∀ i, IsAffine (𝒰.obj i)] :
    IsNoetherian X ↔ ∀ (i : 𝒰.J), IsNoetherianRing Γ(𝒰.obj i, ⊤) := by
  /-
    X : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    inst✝¹ : Finite 𝒰.J
    inst✝ : ∀ (i : 𝒰.J), AlgebraicGeometry.IsAffine (𝒰.obj i)
    ⊢ Iff (AlgebraicGeometry.IsNoetherian X) (∀ (i : 𝒰.J), IsNoetherianRing ↑((𝒰.o …
  -/
  constructor
    /-
      case mp
      X : AlgebraicGeometry.Scheme
      𝒰 : X.OpenCover
      inst✝¹ : Finite 𝒰.J
      inst✝ : ∀ (i : 𝒰.J), AlgebraicGeometry.IsAffine (𝒰.obj i)
      ⊢ AlgebraicGeometry.IsNoetherian X → ∀ (i : 𝒰.J), IsNoetherianRing ↑((𝒰.obj i) …
    -/
  · intro h i
    /-
      case mp
      X : AlgebraicGeometry.Scheme
      𝒰 : X.OpenCover
      inst✝¹ : Finite 𝒰.J
      inst✝ : ∀ (i : 𝒰.J), AlgebraicGeometry.IsAffine (𝒰.obj i)
      h : AlgebraicGeometry.IsNoetherian X
      i : 𝒰.J
      ⊢ IsNoetherianRing ↑((𝒰.obj i).presheaf.obj { unop := Top.top })
    -/
    apply (isLocallyNoetherian_iff_of_affine_openCover _).mp
    /-
      case mp.a
      X : AlgebraicGeometry.Scheme
      𝒰 : X.OpenCover
      inst✝¹ : Finite 𝒰.J
      inst✝ : ∀ (i : 𝒰.J), AlgebraicGeometry.IsAffine (𝒰.obj i)
      h : AlgebraicGeometry.IsNoetherian X
      i : 𝒰.J
      ⊢ AlgebraicGeometry.IsLocallyNoetherian X
    -/
    exact h.toIsLocallyNoetherian
    /-
      🎉 no goals
    -/
    /-
      case mpr
      X : AlgebraicGeometry.Scheme
      𝒰 : X.OpenCover
      inst✝¹ : Finite 𝒰.J
      inst✝ : ∀ (i : 𝒰.J), AlgebraicGeometry.IsAffine (𝒰.obj i)
      ⊢ (∀ (i : 𝒰.J), IsNoetherianRing ↑((𝒰.obj i).presheaf.obj { unop := Top.top }) …
    -/
  · intro hNoeth
    /-
      case mpr
      X : AlgebraicGeometry.Scheme
      𝒰 : X.OpenCover
      inst✝¹ : Finite 𝒰.J
      inst✝ : ∀ (i : 𝒰.J), AlgebraicGeometry.IsAffine (𝒰.obj i)
      hNoeth : ∀ (i : 𝒰.J), IsNoetherianRing ↑((𝒰.obj i).presheaf.obj { unop := Top. …
      ⊢ AlgebraicGeometry.IsNoetherian X
    -/
    convert IsNoetherian.mk
      /-
        case mpr.convert_2
        X : AlgebraicGeometry.Scheme
        𝒰 : X.OpenCover
        inst✝¹ : Finite 𝒰.J
        inst✝ : ∀ (i : 𝒰.J), AlgebraicGeometry.IsAffine (𝒰.obj i)
        hNoeth : ∀ (i : 𝒰.J), IsNoetherianRing ↑((𝒰.obj i).presheaf.obj { unop := Top. …
        ⊢ AlgebraicGeometry.IsLocallyNoetherian X
      -/
    · exact (isLocallyNoetherian_iff_of_affine_openCover _).mpr hNoeth
      /-
        🎉 no goals
      -/
      /-
        case mpr.convert_3
        X : AlgebraicGeometry.Scheme
        𝒰 : X.OpenCover
        inst✝¹ : Finite 𝒰.J
        inst✝ : ∀ (i : 𝒰.J), AlgebraicGeometry.IsAffine (𝒰.obj i)
        hNoeth : ∀ (i : 𝒰.J), IsNoetherianRing ↑((𝒰.obj i).presheaf.obj { unop := Top. …
        ⊢ CompactSpace ↑↑X.toPresheafedSpace
      -/
    · exact Scheme.OpenCover.compactSpace 𝒰
      /-
        🎉 no goals
      -/


open CategoryTheory in
/-- A Noetherian scheme has a Noetherian underlying topological space.

[Stacks, Lemma 01OZ](https://stacks.math.columbia.edu/tag/01OZ) -/
instance (priority := 100) IsNoetherian.noetherianSpace [IsNoetherian X] :
    NoetherianSpace X := by
  /-
    X : AlgebraicGeometry.Scheme
    inst✝ : AlgebraicGeometry.IsNoetherian X
    ⊢ TopologicalSpace.NoetherianSpace ↑↑X.toPresheafedSpace
  -/
  apply TopologicalSpace.noetherian_univ_iff.mp
  /-
    X : AlgebraicGeometry.Scheme
    inst✝ : AlgebraicGeometry.IsNoetherian X
    ⊢ TopologicalSpace.NoetherianSpace ↑Set.univ
  -/
  let 𝒰 := X.affineCover.finiteSubcover
  /-
    X : AlgebraicGeometry.Scheme
    inst✝ : AlgebraicGeometry.IsNoetherian X
    𝒰 : X.OpenCover := X.affineCover.finiteSubcover
    ⊢ TopologicalSpace.NoetherianSpace ↑Set.univ
  -/
  rw [← 𝒰.iUnion_range]
  suffices ∀ i : 𝒰.J, NoetherianSpace (Set.range <| (𝒰.map i).base) by
    apply NoetherianSpace.iUnion
  /-
    X : AlgebraicGeometry.Scheme
    inst✝ : AlgebraicGeometry.IsNoetherian X
    𝒰 : X.OpenCover := X.affineCover.finiteSubcover
    ⊢ ∀ (i : 𝒰.J), TopologicalSpace.NoetherianSpace ↑(Set.range ⇑(𝒰.map i).base)
  -/
  intro i
  have : IsAffine (𝒰.obj i) := by
    rw [X.affineCover.finiteSubcover_obj]
    apply Scheme.isAffine_affineCover
  /-
    X : AlgebraicGeometry.Scheme
    inst✝ : AlgebraicGeometry.IsNoetherian X
    𝒰 : X.OpenCover := X.affineCover.finiteSubcover
    i : 𝒰.J
    this : AlgebraicGeometry.IsAffine (𝒰.obj i)
    ⊢ TopologicalSpace.NoetherianSpace ↑(Set.range ⇑(𝒰.map i).base)
  -/
  let U : X.affineOpens := ⟨Scheme.Hom.opensRange (𝒰.map i), isAffineOpen_opensRange _⟩
  /-
    X : AlgebraicGeometry.Scheme
    inst✝ : AlgebraicGeometry.IsNoetherian X
    𝒰 : X.OpenCover := X.affineCover.finiteSubcover
    i : 𝒰.J
    this : AlgebraicGeometry.IsAffine (𝒰.obj i)
    U : ↑X.affineOpens := ⟨AlgebraicGeometry.Scheme.Hom.opensRange (𝒰.map i), ⋯⟩
    ⊢ TopologicalSpace.NoetherianSpace ↑(Set.range ⇑(𝒰.map i).base)
  -/
  convert noetherianSpace_of_isAffineOpen U.1 U.2
  /-
    X : AlgebraicGeometry.Scheme
    inst✝ : AlgebraicGeometry.IsNoetherian X
    𝒰 : X.OpenCover := X.affineCover.finiteSubcover
    i : 𝒰.J
    this : AlgebraicGeometry.IsAffine (𝒰.obj i)
    U : ↑X.affineOpens := ⟨AlgebraicGeometry.Scheme.Hom.opensRange (𝒰.map i), ⋯⟩
    ⊢ IsNoetherianRing ↑(X.presheaf.obj { unop := ↑U })
  -/
  apply IsLocallyNoetherian.component_noetherian
  /-
    🎉 no goals
  -/


/-- Any morphism of schemes `f : X ⟶ Y` with `X` Noetherian is quasi-compact.

[Stacks, Lemma 01P0](https://stacks.math.columbia.edu/tag/01P0) -/
instance (priority := 100) quasiCompact_of_noetherianSpace_source {X Y : Scheme}
    [NoetherianSpace X] (f : X ⟶ Y) : QuasiCompact f :=
  ⟨fun _ _ _ => NoetherianSpace.isCompact _⟩


/-- If `R` is a Noetherian ring, `Spec R` is a locally Noetherian scheme. -/
instance {R : CommRingCat} [IsNoetherianRing R] :
    IsLocallyNoetherian (Spec R) := by
  apply isLocallyNoetherian_of_affine_cover
    (ι := Fin 1) (S := fun _ => ⟨⊤, isAffineOpen_top (Spec R)⟩)
    /-
      case hS
      X : AlgebraicGeometry.Scheme
      R : CommRingCat
      inst✝ : IsNoetherianRing ↑R
      ⊢ Eq (iSup fun i => ↑⟨Top.top, ⋯⟩) Top.top
    -/
  · exact iSup_const
    /-
      🎉 no goals
    -/
    /-
      case hS'
      X : AlgebraicGeometry.Scheme
      R : CommRingCat
      inst✝ : IsNoetherianRing ↑R
      ⊢ Fin 1 → IsNoetherianRing ↑((AlgebraicGeometry.Spec R).presheaf.obj { unop := …
    -/
  · intro
    /-
      case hS'
      X : AlgebraicGeometry.Scheme
      R : CommRingCat
      inst✝ : IsNoetherianRing ↑R
      i✝ : Fin 1
      ⊢ IsNoetherianRing ↑((AlgebraicGeometry.Spec R).presheaf.obj { unop := ↑⟨Top.t …
    -/
    apply isNoetherianRing_of_ringEquiv R
    /-
      case hS'.f
      X : AlgebraicGeometry.Scheme
      R : CommRingCat
      inst✝ : IsNoetherianRing ↑R
      i✝ : Fin 1
      ⊢ RingEquiv ↑R ↑((AlgebraicGeometry.Spec R).presheaf.obj { unop := ↑⟨Top.top,  …
    -/
    apply CategoryTheory.Iso.commRingCatIsoToRingEquiv
    /-
      case hS'.f.e
      X : AlgebraicGeometry.Scheme
      R : CommRingCat
      inst✝ : IsNoetherianRing ↑R
      i✝ : Fin 1
      ⊢ CategoryTheory.Iso R ((AlgebraicGeometry.Spec R).presheaf.obj { unop := ↑⟨To …
    -/
    exact (Scheme.ΓSpecIso R).symm
    /-
      🎉 no goals
    -/


instance (priority := 100) {R : CommRingCat}
    [IsLocallyNoetherian (Spec R)] : IsNoetherianRing R := by
  /-
    X : AlgebraicGeometry.Scheme
    R : CommRingCat
    inst✝ : AlgebraicGeometry.IsLocallyNoetherian (AlgebraicGeometry.Spec R)
    ⊢ IsNoetherianRing ↑R
  -/
  have := IsLocallyNoetherian.component_noetherian ⟨⊤, AlgebraicGeometry.isAffineOpen_top (Spec R)⟩
  /-
    X : AlgebraicGeometry.Scheme
    R : CommRingCat
    inst✝ : AlgebraicGeometry.IsLocallyNoetherian (AlgebraicGeometry.Spec R)
    this : IsNoetherianRing ↑((AlgebraicGeometry.Spec R).presheaf.obj { unop := ↑⟨ …
    ⊢ IsNoetherianRing ↑R
  -/
  apply isNoetherianRing_of_ringEquiv Γ(Spec R, ⊤)
  /-
    case f
    X : AlgebraicGeometry.Scheme
    R : CommRingCat
    inst✝ : AlgebraicGeometry.IsLocallyNoetherian (AlgebraicGeometry.Spec R)
    this : IsNoetherianRing ↑((AlgebraicGeometry.Spec R).presheaf.obj { unop := ↑⟨ …
    ⊢ RingEquiv ↑((AlgebraicGeometry.Spec R).presheaf.obj { unop := Top.top }) ↑R
  -/
  apply CategoryTheory.Iso.commRingCatIsoToRingEquiv
  /-
    case f.e
    X : AlgebraicGeometry.Scheme
    R : CommRingCat
    inst✝ : AlgebraicGeometry.IsLocallyNoetherian (AlgebraicGeometry.Spec R)
    this : IsNoetherianRing ↑((AlgebraicGeometry.Spec R).presheaf.obj { unop := ↑⟨ …
    ⊢ CategoryTheory.Iso ((AlgebraicGeometry.Spec R).presheaf.obj { unop := Top.to …
  -/
  exact Scheme.ΓSpecIso R
  /-
    🎉 no goals
  -/


/-- If `R` is a Noetherian ring, `Spec R` is a Noetherian scheme. -/
instance {R : CommRingCat} [IsNoetherianRing R] :
    IsNoetherian (Spec R) where


instance {R} [CommRing R] [IsNoetherianRing R] :
    IsNoetherian (Spec (.of R)) := by
  /-
    X : AlgebraicGeometry.Scheme
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsNoetherianRing R
    ⊢ AlgebraicGeometry.IsNoetherian (AlgebraicGeometry.Spec (CommRingCat.of R))
  -/
  suffices IsNoetherianRing (CommRingCat.of R) by infer_instance
  /-
    X : AlgebraicGeometry.Scheme
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsNoetherianRing R
    ⊢ IsNoetherianRing ↑(CommRingCat.of R)
  -/
  simp only [CommRingCat.coe_of]
  /-
    X : AlgebraicGeometry.Scheme
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsNoetherianRing R
    ⊢ IsNoetherianRing R
  -/
  assumption
  /-
    🎉 no goals
  -/


/-- `R` is a Noetherian ring if and only if `Spec R` is a Noetherian scheme. -/
theorem isNoetherian_Spec {R : CommRingCat} :
    IsNoetherian (Spec R) ↔ IsNoetherianRing R :=
  ⟨fun _ => inferInstance,
   fun _ => inferInstance⟩


/-- A Noetherian scheme has a finite number of irreducible components.

[Stacks, Lemma 0BA8](https://stacks.math.columbia.edu/tag/0BA8) -/
theorem finite_irreducibleComponents_of_isNoetherian [IsNoetherian X] :
    (irreducibleComponents X).Finite := NoetherianSpace.finite_irreducibleComponents


