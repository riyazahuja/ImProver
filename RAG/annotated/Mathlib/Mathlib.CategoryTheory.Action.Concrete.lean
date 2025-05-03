/-- Bundles a type `H` with a multiplicative action of `G` as an `Action`. -/
def ofMulAction (G H : Type u) [Monoid G] [MulAction G H] : Action (Type u) (MonCat.of G) where
  V := H
                                     /-
                                       G H : Type u
                                       inst✝¹ : Monoid G
                                       inst✝ : MulAction G H
                                       ⊢ MulAction (↑(MonCat.of G)) H
                                     -/
  ρ := @MulAction.toEndHom _ _ _ (by assumption)
                                     /-
                                       🎉 no goals
                                     -/


@[simp]
theorem ofMulAction_apply {G H : Type u} [Monoid G] [MulAction G H] (g : G) (x : H) :
    (ofMulAction G H).ρ g x = (g • x : H) :=
  rfl


/-- Given a family `F` of types with `G`-actions, this is the limit cone demonstrating that the
product of `F` as types is a product in the category of `G`-sets. -/
def ofMulActionLimitCone {ι : Type v} (G : Type max v u) [Monoid G] (F : ι → Type max v u)
    [∀ i : ι, MulAction G (F i)] :
    LimitCone (Discrete.functor fun i : ι => Action.ofMulAction G (F i)) where
  cone :=
    { pt := Action.ofMulAction G (∀ i : ι, F i)
      π := Discrete.natTrans (fun i => ⟨fun x => x i.as, fun _ => rfl⟩) }
  isLimit :=
    { lift := fun s =>
        { hom := fun x i => (s.π.app ⟨i⟩).hom x
          comm := fun g => by
            /-
              ι : Type v
              G : Type (max v u)
              inst✝¹ : Monoid G
              F : ι → Type (max v u)
              inst✝ : (i : ι) → MulAction G (F i)
              s : CategoryTheory.Limits.Cone (CategoryTheory.Discrete.functor fun i => Actio …
              g : ↑(MonCat.of G)
              ⊢ Eq (CategoryTheory.CategoryStruct.comp (s.pt.ρ g) fun x i => (s.π.app { as : …
            -/
            ext x
            /-
              case h
              ι : Type v
              G : Type (max v u)
              inst✝¹ : Monoid G
              F : ι → Type (max v u)
              inst✝ : (i : ι) → MulAction G (F i)
              s : CategoryTheory.Limits.Cone (CategoryTheory.Discrete.functor fun i => Actio …
              g : ↑(MonCat.of G)
              x : s.pt.V
              ⊢ Eq (CategoryTheory.CategoryStruct.comp (s.pt.ρ g) (fun x i => (s.π.app { as  …
            -/
            funext j
            /-
              case h.h
              ι : Type v
              G : Type (max v u)
              inst✝¹ : Monoid G
              F : ι → Type (max v u)
              inst✝ : (i : ι) → MulAction G (F i)
              s : CategoryTheory.Limits.Cone (CategoryTheory.Discrete.functor fun i => Actio …
              g : ↑(MonCat.of G)
              x : s.pt.V
              j : ι
              ⊢ Eq (CategoryTheory.CategoryStruct.comp (s.pt.ρ g) (fun x i => (s.π.app { as  …
            -/
            exact congr_fun ((s.π.app ⟨j⟩).comm g) x }
            /-
              🎉 no goals
            -/
      fac := fun _ _ => rfl
      uniq := fun s f h => by
        /-
          ι : Type v
          G : Type (max v u)
          inst✝¹ : Monoid G
          F : ι → Type (max v u)
          inst✝ : (i : ι) → MulAction G (F i)
          s : CategoryTheory.Limits.Cone (CategoryTheory.Discrete.functor fun i => Actio …
          f : Quiver.Hom s.pt { pt := Action.ofMulAction G ((i : ι) → F i), π := Categor …
          h : ∀ (j : CategoryTheory.Discrete ι), Eq (CategoryTheory.CategoryStruct.comp  …
          ⊢ Eq f ((fun s => { hom := fun x i => (s.π.app { as := i }).hom x, comm := ⋯ } …
        -/
        ext x
        /-
          case h.h
          ι : Type v
          G : Type (max v u)
          inst✝¹ : Monoid G
          F : ι → Type (max v u)
          inst✝ : (i : ι) → MulAction G (F i)
          s : CategoryTheory.Limits.Cone (CategoryTheory.Discrete.functor fun i => Actio …
          f : Quiver.Hom s.pt { pt := Action.ofMulAction G ((i : ι) → F i), π := Categor …
          h : ∀ (j : CategoryTheory.Discrete ι), Eq (CategoryTheory.CategoryStruct.comp  …
          x : s.pt.V
          ⊢ Eq (f.hom x) (((fun s => { hom := fun x i => (s.π.app { as := i }).hom x, co …
        -/
        funext j
        /-
          case h.h.h
          ι : Type v
          G : Type (max v u)
          inst✝¹ : Monoid G
          F : ι → Type (max v u)
          inst✝ : (i : ι) → MulAction G (F i)
          s : CategoryTheory.Limits.Cone (CategoryTheory.Discrete.functor fun i => Actio …
          f : Quiver.Hom s.pt { pt := Action.ofMulAction G ((i : ι) → F i), π := Categor …
          h : ∀ (j : CategoryTheory.Discrete ι), Eq (CategoryTheory.CategoryStruct.comp  …
          x : s.pt.V
          j : ι
          ⊢ Eq (f.hom x j) (((fun s => { hom := fun x i => (s.π.app { as := i }).hom x,  …
        -/
        dsimp at *
        /-
          case h.h.h
          ι : Type v
          G : Type (max v u)
          inst✝¹ : Monoid G
          F : ι → Type (max v u)
          inst✝ : (i : ι) → MulAction G (F i)
          s : CategoryTheory.Limits.Cone (CategoryTheory.Discrete.functor fun i => Actio …
          f : Quiver.Hom s.pt { pt := Action.ofMulAction G ((i : ι) → F i), π := Categor …
          h : ∀ (j : CategoryTheory.Discrete ι), Eq (CategoryTheory.CategoryStruct.comp  …
          x : s.pt.V
          j : ι
          ⊢ Eq (f.hom x j) ((s.π.app { as := j }).hom x)
        -/
        rw [← h ⟨j⟩]
        /-
          case h.h.h
          ι : Type v
          G : Type (max v u)
          inst✝¹ : Monoid G
          F : ι → Type (max v u)
          inst✝ : (i : ι) → MulAction G (F i)
          s : CategoryTheory.Limits.Cone (CategoryTheory.Discrete.functor fun i => Actio …
          f : Quiver.Hom s.pt { pt := Action.ofMulAction G ((i : ι) → F i), π := Categor …
          h : ∀ (j : CategoryTheory.Discrete ι), Eq (CategoryTheory.CategoryStruct.comp  …
          x : s.pt.V
          j : ι
          ⊢ Eq (f.hom x j) ((CategoryTheory.CategoryStruct.comp f { hom := fun x => x {  …
        -/
        rfl }
        /-
          🎉 no goals
        -/


/-- The `G`-set `G`, acting on itself by left multiplication. -/
@[simps!]
def leftRegular (G : Type u) [Monoid G] : Action (Type u) (MonCat.of G) :=
  Action.ofMulAction G G


/-- The `G`-set `Gⁿ`, acting on itself by left multiplication. -/
@[simps!]
def diagonal (G : Type u) [Monoid G] (n : ℕ) : Action (Type u) (MonCat.of G) :=
  Action.ofMulAction G (Fin n → G)


/-- We have `Fin 1 → G ≅ G` as `G`-sets, with `G` acting by left multiplication. -/
def diagonalOneIsoLeftRegular (G : Type u) [Monoid G] : diagonal G 1 ≅ leftRegular G :=
  Action.mkIso (Equiv.funUnique _ _).toIso fun _ => rfl


/-- If `X` is a type with `[Fintype X]` and `G` acts on `X`, then `G` also acts on
`FintypeCat.of X`. -/
instance (G : Type*) (X : Type*) [Monoid G] [MulAction G X] [Fintype X] :
    MulAction G (FintypeCat.of X) :=
  inferInstanceAs <| MulAction G X


/-- Bundles a finite type `H` with a multiplicative action of `G` as an `Action`. -/
def ofMulAction (G : Type u) (H : FintypeCat.{u}) [Monoid G] [MulAction G H] :
    Action FintypeCat (MonCat.of G) where
  V := H
                                     /-
                                       G : Type u
                                       H : FintypeCat
                                       inst✝¹ : Monoid G
                                       inst✝ : MulAction G ↑H
                                       ⊢ MulAction ↑(MonCat.of G) ↑H
                                     -/
  ρ := @MulAction.toEndHom _ _ _ (by assumption)
                                     /-
                                       🎉 no goals
                                     -/


@[simp]
theorem ofMulAction_apply {G : Type u} {H : FintypeCat.{u}} [Monoid G] [MulAction G H]
    (g : G) (x : H) : (FintypeCat.ofMulAction G H).ρ g x = (g • x : H) :=
  rfl


/-- Shorthand notation for the quotient of `G` by `H` as a finite `G`-set. -/
notation:10 G:10 " ⧸ₐ " H:10 => Action.FintypeCat.ofMulAction G (FintypeCat.of <| G ⧸ H)


/-- If `N` is a normal subgroup of `G`, then this is the group homomorphism
sending an element `g` of `G` to the `G`-endomorphism of `G ⧸ₐ N` given by
multiplication with `g⁻¹` on the right. -/
def toEndHom [N.Normal] : G →* End (G ⧸ₐ N) where
  toFun v := {
    hom := Quotient.lift (fun σ ↦ ⟦σ * v⁻¹⟧) <| fun a b h ↦ Quotient.sound <| by
      /-
        G : Type u_1
        inst✝² : Group G
        H N : Subgroup G
        inst✝¹ : Fintype (HasQuotient.Quotient G N)
        inst✝ : N.Normal
        v a b : G
        h : HasEquiv.Equiv a b
        ⊢ HasEquiv.Equiv (HMul.hMul a (Inv.inv v)) (HMul.hMul b (Inv.inv v))
      -/
      apply (QuotientGroup.leftRel_apply).mpr
      /-
        G : Type u_1
        inst✝² : Group G
        H N : Subgroup G
        inst✝¹ : Fintype (HasQuotient.Quotient G N)
        inst✝ : N.Normal
        v a b : G
        h : HasEquiv.Equiv a b
        ⊢ Membership.mem N (HMul.hMul (Inv.inv (HMul.hMul a (Inv.inv v))) (HMul.hMul b …
      -/
      simp only [mul_inv_rev, inv_inv]
      /-
        G : Type u_1
        inst✝² : Group G
        H N : Subgroup G
        inst✝¹ : Fintype (HasQuotient.Quotient G N)
        inst✝ : N.Normal
        v a b : G
        h : HasEquiv.Equiv a b
        ⊢ Membership.mem N (HMul.hMul (HMul.hMul v (Inv.inv a)) (HMul.hMul b (Inv.inv  …
      -/
      convert_to v * (a⁻¹ * b) * v⁻¹ ∈ N
        /-
          case h.e'_5
          G : Type u_1
          inst✝² : Group G
          H N : Subgroup G
          inst✝¹ : Fintype (HasQuotient.Quotient G N)
          inst✝ : N.Normal
          v a b : G
          h : HasEquiv.Equiv a b
          ⊢ Eq (HMul.hMul (HMul.hMul v (Inv.inv a)) (HMul.hMul b (Inv.inv v))) (HMul.hMu …
        -/
      · group
        /-
          🎉 no goals
        -/
        /-
          G : Type u_1
          inst✝² : Group G
          H N : Subgroup G
          inst✝¹ : Fintype (HasQuotient.Quotient G N)
          inst✝ : N.Normal
          v a b : G
          h : HasEquiv.Equiv a b
          ⊢ Membership.mem N (HMul.hMul (HMul.hMul v (HMul.hMul (Inv.inv a) b)) (Inv.inv …
        -/
      · exact Subgroup.Normal.conj_mem ‹_› _ (QuotientGroup.leftRel_apply.mp h) _
        /-
          🎉 no goals
        -/
    comm := fun (g : G) ↦ by
      /-
        G : Type u_1
        inst✝² : Group G
        H N : Subgroup G
        inst✝¹ : Fintype (HasQuotient.Quotient G N)
        inst✝ : N.Normal
        v g : G
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((Action.FintypeCat.ofMulAction G (Fi …
      -/
      ext (x : G ⧸ N)
      /-
        case h
        G : Type u_1
        inst✝² : Group G
        H N : Subgroup G
        inst✝¹ : Fintype (HasQuotient.Quotient G N)
        inst✝ : N.Normal
        v g : G
        x : HasQuotient.Quotient G N
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((Action.FintypeCat.ofMulAction G (Fi …
      -/
      induction' x using Quotient.inductionOn with x
      /-
        case h.h
        G : Type u_1
        inst✝² : Group G
        H N : Subgroup G
        inst✝¹ : Fintype (HasQuotient.Quotient G N)
        inst✝ : N.Normal
        v g x : G
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((Action.FintypeCat.ofMulAction G (Fi …
      -/
      simp only [FintypeCat.comp_apply, Action.FintypeCat.ofMulAction_apply, Quotient.lift_mk]
      /-
        case h.h
        G : Type u_1
        inst✝² : Group G
        H N : Subgroup G
        inst✝¹ : Fintype (HasQuotient.Quotient G N)
        inst✝ : N.Normal
        v g x : G
        ⊢ Eq (Quotient.lift (fun σ => Quotient.mk (QuotientGroup.leftRel N) (HMul.hMul …
      -/
      show Quotient.lift (fun σ ↦ ⟦σ * v⁻¹⟧) _ (⟦g • x⟧) = _
      /-
        case h.h
        G : Type u_1
        inst✝² : Group G
        H N : Subgroup G
        inst✝¹ : Fintype (HasQuotient.Quotient G N)
        inst✝ : N.Normal
        v g x : G
        ⊢ Eq (Quotient.lift (fun σ => Quotient.mk (QuotientGroup.leftRel N) (HMul.hMul …
      -/
      simp only [smul_eq_mul, Quotient.lift_mk, mul_assoc]
      /-
        case h.h
        G : Type u_1
        inst✝² : Group G
        H N : Subgroup G
        inst✝¹ : Fintype (HasQuotient.Quotient G N)
        inst✝ : N.Normal
        v g x : G
        ⊢ Eq (Quotient.mk (QuotientGroup.leftRel N) (HMul.hMul g (HMul.hMul x (Inv.inv …
      -/
      rfl
      /-
        🎉 no goals
      -/
  }
  map_one' := by
    /-
      G : Type u_1
      inst✝² : Group G
      H N : Subgroup G
      inst✝¹ : Fintype (HasQuotient.Quotient G N)
      inst✝ : N.Normal
      ⊢ Eq ((fun v => { hom := Quotient.lift (fun σ => Quotient.mk (QuotientGroup.le …
    -/
    apply Action.hom_ext
    /-
      case h
      G : Type u_1
      inst✝² : Group G
      H N : Subgroup G
      inst✝¹ : Fintype (HasQuotient.Quotient G N)
      inst✝ : N.Normal
      ⊢ Eq ((fun v => { hom := Quotient.lift (fun σ => Quotient.mk (QuotientGroup.le …
    -/
    ext (x : G ⧸ N)
    /-
      case h.h
      G : Type u_1
      inst✝² : Group G
      H N : Subgroup G
      inst✝¹ : Fintype (HasQuotient.Quotient G N)
      inst✝ : N.Normal
      x : HasQuotient.Quotient G N
      ⊢ Eq (((fun v => { hom := Quotient.lift (fun σ => Quotient.mk (QuotientGroup.l …
    -/
    induction' x using Quotient.inductionOn with x
    /-
      case h.h.h
      G : Type u_1
      inst✝² : Group G
      H N : Subgroup G
      inst✝¹ : Fintype (HasQuotient.Quotient G N)
      inst✝ : N.Normal
      x : G
      ⊢ Eq (((fun v => { hom := Quotient.lift (fun σ => Quotient.mk (QuotientGroup.l …
    -/
    simp
    /-
      🎉 no goals
    -/
  map_mul' σ τ := by
    /-
      G : Type u_1
      inst✝² : Group G
      H N : Subgroup G
      inst✝¹ : Fintype (HasQuotient.Quotient G N)
      inst✝ : N.Normal
      σ τ : G
      ⊢ Eq ({ toFun := fun v => { hom := Quotient.lift (fun σ => Quotient.mk (Quotie …
    -/
    apply Action.hom_ext
    /-
      case h
      G : Type u_1
      inst✝² : Group G
      H N : Subgroup G
      inst✝¹ : Fintype (HasQuotient.Quotient G N)
      inst✝ : N.Normal
      σ τ : G
      ⊢ Eq ({ toFun := fun v => { hom := Quotient.lift (fun σ => Quotient.mk (Quotie …
    -/
    ext (x : G ⧸ N)
    /-
      case h.h
      G : Type u_1
      inst✝² : Group G
      H N : Subgroup G
      inst✝¹ : Fintype (HasQuotient.Quotient G N)
      inst✝ : N.Normal
      σ τ : G
      x : HasQuotient.Quotient G N
      ⊢ Eq (({ toFun := fun v => { hom := Quotient.lift (fun σ => Quotient.mk (Quoti …
    -/
    induction' x using Quotient.inductionOn with x
    /-
      case h.h.h
      G : Type u_1
      inst✝² : Group G
      H N : Subgroup G
      inst✝¹ : Fintype (HasQuotient.Quotient G N)
      inst✝ : N.Normal
      σ τ x : G
      ⊢ Eq (({ toFun := fun v => { hom := Quotient.lift (fun σ => Quotient.mk (Quoti …
    -/
    show ⟦x * (σ * τ)⁻¹⟧ = ⟦x * τ⁻¹ * σ⁻¹⟧
    /-
      case h.h.h
      G : Type u_1
      inst✝² : Group G
      H N : Subgroup G
      inst✝¹ : Fintype (HasQuotient.Quotient G N)
      inst✝ : N.Normal
      σ τ x : G
      ⊢ Eq (Quotient.mk (QuotientGroup.leftRel N) (HMul.hMul x (Inv.inv (HMul.hMul σ …
    -/
    rw [mul_inv_rev, mul_assoc]
    /-
      🎉 no goals
    -/


@[simp]
lemma toEndHom_apply [N.Normal] (g h : G) : (toEndHom N g).hom ⟦h⟧ = ⟦h * g⁻¹⟧ := rfl


variable {N} in
lemma toEndHom_trivial_of_mem [N.Normal] {n : G} (hn : n ∈ N) : toEndHom N n = 𝟙 (G ⧸ₐ N) := by
  /-
    G : Type u_1
    inst✝² : Group G
    N : Subgroup G
    inst✝¹ : Fintype (HasQuotient.Quotient G N)
    inst✝ : N.Normal
    n : G
    hn : Membership.mem N n
    ⊢ Eq ((Action.FintypeCat.toEndHom N) n) (CategoryTheory.CategoryStruct.id (Act …
  -/
  apply Action.hom_ext
  /-
    case h
    G : Type u_1
    inst✝² : Group G
    N : Subgroup G
    inst✝¹ : Fintype (HasQuotient.Quotient G N)
    inst✝ : N.Normal
    n : G
    hn : Membership.mem N n
    ⊢ Eq ((Action.FintypeCat.toEndHom N) n).hom (CategoryTheory.CategoryStruct.id  …
  -/
  ext (x : G ⧸ N)
  /-
    case h.h
    G : Type u_1
    inst✝² : Group G
    N : Subgroup G
    inst✝¹ : Fintype (HasQuotient.Quotient G N)
    inst✝ : N.Normal
    n : G
    hn : Membership.mem N n
    x : HasQuotient.Quotient G N
    ⊢ Eq (((Action.FintypeCat.toEndHom N) n).hom x) ((CategoryTheory.CategoryStruc …
  -/
  induction' x using Quotient.inductionOn with μ
  /-
    case h.h.h
    G : Type u_1
    inst✝² : Group G
    N : Subgroup G
    inst✝¹ : Fintype (HasQuotient.Quotient G N)
    inst✝ : N.Normal
    n : G
    hn : Membership.mem N n
    μ : G
    ⊢ Eq (((Action.FintypeCat.toEndHom N) n).hom (Quotient.mk (QuotientGroup.leftR …
  -/
  exact Quotient.sound ((QuotientGroup.leftRel_apply).mpr <| by simpa)
  /-
    🎉 no goals
  -/


/-- If `H` and `N` are subgroups of a group `G` with `N` normal, there is a canonical
group homomorphism `H ⧸ N ⊓ H` to the `G`-endomorphisms of `G ⧸ N`. -/
def quotientToEndHom [N.Normal] : H ⧸ Subgroup.subgroupOf N H →* End (G ⧸ₐ N) :=
  QuotientGroup.lift (Subgroup.subgroupOf N H) ((toEndHom N).comp H.subtype) <| fun _ uinU' ↦
    toEndHom_trivial_of_mem uinU'


@[simp]
lemma quotientToEndHom_mk [N.Normal] (x : H) (g : G) :
    (quotientToEndHom H N ⟦x⟧).hom ⟦g⟧ = ⟦g * x⁻¹⟧ :=
  rfl


/-- If `N` and `H` are subgroups of a group `G` with `N ≤ H`, this is the canonical
`G`-morphism `G ⧸ N ⟶ G ⧸ H`. -/
def quotientToQuotientOfLE [Fintype (G ⧸ H)] (h : N ≤ H) : (G ⧸ₐ N) ⟶ (G ⧸ₐ H) where
  hom := Quotient.lift _ <| fun _ _ hab ↦ Quotient.sound <|
    (QuotientGroup.leftRel_apply).mpr (h <| (QuotientGroup.leftRel_apply).mp hab)
  comm g := by
    /-
      G : Type u_1
      inst✝² : Group G
      H N : Subgroup G
      inst✝¹ : Fintype (HasQuotient.Quotient G N)
      inst✝ : Fintype (HasQuotient.Quotient G H)
      h : LE.le N H
      g : ↑(MonCat.of G)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((Action.FintypeCat.ofMulAction G (Fi …
    -/
    ext (x : G ⧸ N)
    /-
      case h
      G : Type u_1
      inst✝² : Group G
      H N : Subgroup G
      inst✝¹ : Fintype (HasQuotient.Quotient G N)
      inst✝ : Fintype (HasQuotient.Quotient G H)
      h : LE.le N H
      g : ↑(MonCat.of G)
      x : HasQuotient.Quotient G N
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((Action.FintypeCat.ofMulAction G (Fi …
    -/
    induction' x using Quotient.inductionOn with μ
    /-
      case h.h
      G : Type u_1
      inst✝² : Group G
      H N : Subgroup G
      inst✝¹ : Fintype (HasQuotient.Quotient G N)
      inst✝ : Fintype (HasQuotient.Quotient G H)
      h : LE.le N H
      g : ↑(MonCat.of G)
      μ : G
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((Action.FintypeCat.ofMulAction G (Fi …
    -/
    rfl
    /-
      🎉 no goals
    -/


@[simp]
lemma quotientToQuotientOfLE_hom_mk [Fintype (G ⧸ H)] (h : N ≤ H) (x : G) :
    (quotientToQuotientOfLE H N h).hom ⟦x⟧ = ⟦x⟧ :=
  rfl


instance instMulAction {G : MonCat.{u}} (X : Action V G) :
    MulAction G ((CategoryTheory.forget _).obj X) where
  smul g x := ((CategoryTheory.forget _).map (X.ρ g)) x
  one_smul x := by
    /-
      V : Type (u + 1)
      inst✝¹ : CategoryTheory.LargeCategory V
      inst✝ : CategoryTheory.ConcreteCategory V
      G : MonCat
      X : Action V G
      x : (CategoryTheory.forget (Action V G)).obj X
      ⊢ Eq (HSMul.hSMul 1 x) x
    -/
    show ((CategoryTheory.forget _).map (X.ρ 1)) x = x
    /-
      V : Type (u + 1)
      inst✝¹ : CategoryTheory.LargeCategory V
      inst✝ : CategoryTheory.ConcreteCategory V
      G : MonCat
      X : Action V G
      x : (CategoryTheory.forget (Action V G)).obj X
      ⊢ Eq ((CategoryTheory.forget V).map (X.ρ 1) x) x
    -/
    simp only [Action.ρ_one, FunctorToTypes.map_id_apply]
    /-
      🎉 no goals
    -/
  mul_smul g h x := by
    show (CategoryTheory.forget V).map (X.ρ (g * h)) x =
      ((CategoryTheory.forget V).map (X.ρ h) ≫ (CategoryTheory.forget V).map (X.ρ g)) x
    /-
      V : Type (u + 1)
      inst✝¹ : CategoryTheory.LargeCategory V
      inst✝ : CategoryTheory.ConcreteCategory V
      G : MonCat
      X : Action V G
      g h : ↑G
      x : (CategoryTheory.forget (Action V G)).obj X
      ⊢ Eq ((CategoryTheory.forget V).map (X.ρ (HMul.hMul g h)) x) (CategoryTheory.C …
    -/
    rw [← Functor.map_comp, map_mul]
    /-
      V : Type (u + 1)
      inst✝¹ : CategoryTheory.LargeCategory V
      inst✝ : CategoryTheory.ConcreteCategory V
      G : MonCat
      X : Action V G
      g h : ↑G
      x : (CategoryTheory.forget (Action V G)).obj X
      ⊢ Eq ((CategoryTheory.forget V).map (HMul.hMul (X.ρ g) (X.ρ h)) x) ((CategoryT …
    -/
    rfl
    /-
      🎉 no goals
    -/

/- Specialize `instMulAction` to assist typeclass inference. -/

instance {G : MonCat.{u}} (X : Action FintypeCat G) : MulAction G X.V := Action.instMulAction X

instance {G : Type u} [Group G] (X : Action FintypeCat (MonCat.of G)) : MulAction G X.V :=
  Action.instMulAction X


