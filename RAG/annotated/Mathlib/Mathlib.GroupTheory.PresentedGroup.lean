/-- Given a set of relations, `rels`, over a type `α`, `PresentedGroup` constructs the group with
generators `x : α` and relations `rels` as a quotient of `FreeGroup α`. -/
def PresentedGroup (rels : Set (FreeGroup α)) :=
  FreeGroup α ⧸ Subgroup.normalClosure rels


instance (rels : Set (FreeGroup α)) : Group (PresentedGroup rels) :=
  QuotientGroup.Quotient.group _


/-- The canonical map from the free group on `α` to a presented group with generators `x : α`,
where `x` is mapped to its equivalence class under the given set of relations `rels`-/
def mk (rels : Set (FreeGroup α)) : FreeGroup α →* PresentedGroup rels :=
  ⟨⟨QuotientGroup.mk, rfl⟩, fun _ _ => rfl⟩


theorem mk_surjective (rels : Set (FreeGroup α)) : Function.Surjective <| mk rels :=
  QuotientGroup.mk_surjective


/-- `of` is the canonical map from `α` to a presented group with generators `x : α`. The term `x` is
mapped to the equivalence class of the image of `x` in `FreeGroup α`. -/
def of {rels : Set (FreeGroup α)} (x : α) : PresentedGroup rels :=
  mk rels (FreeGroup.of x)


/-- The generators of a presented group generate the presented group. That is, the subgroup closure
of the set of generators equals `⊤`. -/
@[simp]
theorem closure_range_of (rels : Set (FreeGroup α)) :
    Subgroup.closure (Set.range (PresentedGroup.of : α → PresentedGroup rels)) = ⊤ := by
  /-
    α : Type u_1
    rels : Set (FreeGroup α)
    ⊢ Eq (Subgroup.closure (Set.range PresentedGroup.of)) Top.top
  -/
  have : (PresentedGroup.of : α → PresentedGroup rels) = QuotientGroup.mk' _ ∘ FreeGroup.of := rfl
  rw [this, Set.range_comp, ← MonoidHom.map_closure (QuotientGroup.mk' _),
    FreeGroup.closure_range_of, ← MonoidHom.range_eq_map]
  /-
    α : Type u_1
    rels : Set (FreeGroup α)
    this : Eq PresentedGroup.of (Function.comp (⇑(QuotientGroup.mk' (Subgroup.norm …
    ⊢ Eq (QuotientGroup.mk' (Subgroup.normalClosure rels)).range Top.top
  -/
  exact MonoidHom.range_eq_top.2 (QuotientGroup.mk'_surjective _)
  /-
    🎉 no goals
  -/


@[induction_eliminator]
theorem induction_on {rels : Set (FreeGroup α)} {C : PresentedGroup rels → Prop}
    (x : PresentedGroup rels) (H : ∀ z, C (mk rels z)) : C x :=
  Quotient.inductionOn' x H


theorem generated_by (rels : Set (FreeGroup α)) (H : Subgroup (PresentedGroup rels))
    (h : ∀ j : α, PresentedGroup.of j ∈ H) (x : PresentedGroup rels) : x ∈ H := by
  /-
    α : Type u_1
    rels : Set (FreeGroup α)
    H : Subgroup (PresentedGroup rels)
    h : ∀ (j : α), Membership.mem H (PresentedGroup.of j)
    x : PresentedGroup rels
    ⊢ Membership.mem H x
  -/
  induction' x with z
  /-
    case H
    α : Type u_1
    rels : Set (FreeGroup α)
    H : Subgroup (PresentedGroup rels)
    h : ∀ (j : α), Membership.mem H (PresentedGroup.of j)
    z : FreeGroup α
    ⊢ Membership.mem H ((PresentedGroup.mk rels) z)
  -/
  induction z
    /-
      case H.C1
      α : Type u_1
      rels : Set (FreeGroup α)
      H : Subgroup (PresentedGroup rels)
      h : ∀ (j : α), Membership.mem H (PresentedGroup.of j)
      ⊢ Membership.mem H ((PresentedGroup.mk rels) 1)
    -/
  · exact one_mem H
    /-
      🎉 no goals
    -/
    /-
      case H.Cp
      α : Type u_1
      rels : Set (FreeGroup α)
      H : Subgroup (PresentedGroup rels)
      h : ∀ (j : α), Membership.mem H (PresentedGroup.of j)
      x✝ : α
      ⊢ Membership.mem H ((PresentedGroup.mk rels) (Pure.pure x✝))
    -/
  · exact h _
    /-
      🎉 no goals
    -/
    /-
      case H.Ci
      α : Type u_1
      rels : Set (FreeGroup α)
      H : Subgroup (PresentedGroup rels)
      h : ∀ (j : α), Membership.mem H (PresentedGroup.of j)
      x✝ : α
      a✝ : Membership.mem H ((PresentedGroup.mk rels) (Pure.pure x✝))
      ⊢ Membership.mem H ((PresentedGroup.mk rels) (Inv.inv (Pure.pure x✝)))
    -/
  · exact (Subgroup.inv_mem_iff H).mpr (by assumption)
    /-
      🎉 no goals
    -/
  /-
    case H.Cm
    α : Type u_1
    rels : Set (FreeGroup α)
    H : Subgroup (PresentedGroup rels)
    h : ∀ (j : α), Membership.mem H (PresentedGroup.of j)
    x✝ y✝ : FreeGroup α
    a✝¹ : Membership.mem H ((PresentedGroup.mk rels) x✝)
    a✝ : Membership.mem H ((PresentedGroup.mk rels) y✝)
    ⊢ Membership.mem H ((PresentedGroup.mk rels) (HMul.hMul x✝ y✝))
  -/
  rename_i h1 h2
  /-
    case H.Cm
    α : Type u_1
    rels : Set (FreeGroup α)
    H : Subgroup (PresentedGroup rels)
    h : ∀ (j : α), Membership.mem H (PresentedGroup.of j)
    x✝ y✝ : FreeGroup α
    h1 : Membership.mem H ((PresentedGroup.mk rels) x✝)
    h2 : Membership.mem H ((PresentedGroup.mk rels) y✝)
    ⊢ Membership.mem H ((PresentedGroup.mk rels) (HMul.hMul x✝ y✝))
  -/
  change QuotientGroup.mk _ ∈ H.carrier
  /-
    case H.Cm
    α : Type u_1
    rels : Set (FreeGroup α)
    H : Subgroup (PresentedGroup rels)
    h : ∀ (j : α), Membership.mem H (PresentedGroup.of j)
    x✝ y✝ : FreeGroup α
    h1 : Membership.mem H ((PresentedGroup.mk rels) x✝)
    h2 : Membership.mem H ((PresentedGroup.mk rels) y✝)
    ⊢ Membership.mem H.carrier ↑(HMul.hMul x✝ y✝)
  -/
  rw [QuotientGroup.mk_mul]
  /-
    case H.Cm
    α : Type u_1
    rels : Set (FreeGroup α)
    H : Subgroup (PresentedGroup rels)
    h : ∀ (j : α), Membership.mem H (PresentedGroup.of j)
    x✝ y✝ : FreeGroup α
    h1 : Membership.mem H ((PresentedGroup.mk rels) x✝)
    h2 : Membership.mem H ((PresentedGroup.mk rels) y✝)
    ⊢ Membership.mem H.carrier (HMul.hMul ↑x✝ ↑y✝)
  -/
  exact Subgroup.mul_mem _ h1 h2
  /-
    🎉 no goals
  -/


local notation "F" => FreeGroup.lift f


theorem closure_rels_subset_ker (h : ∀ r ∈ rels, FreeGroup.lift f r = 1) :
    Subgroup.normalClosure rels ≤ MonoidHom.ker F :=
  Subgroup.normalClosure_le_normal fun x w ↦ MonoidHom.mem_ker.2 (h x w)


theorem to_group_eq_one_of_mem_closure (h : ∀ r ∈ rels, FreeGroup.lift f r = 1) :
    ∀ x ∈ Subgroup.normalClosure rels, F x = 1 :=
  fun _ w ↦ MonoidHom.mem_ker.1 <| closure_rels_subset_ker h w


/-- The extension of a map `f : α → G` that satisfies the given relations to a group homomorphism
from `PresentedGroup rels → G`. -/
def toGroup (h : ∀ r ∈ rels, FreeGroup.lift f r = 1) : PresentedGroup rels →* G :=
  QuotientGroup.lift (Subgroup.normalClosure rels) F (to_group_eq_one_of_mem_closure h)


@[simp]
theorem toGroup.of (h : ∀ r ∈ rels, FreeGroup.lift f r = 1) {x : α} : toGroup h (of x) = f x :=
  FreeGroup.lift.of


theorem toGroup.unique (h : ∀ r ∈ rels, FreeGroup.lift f r = 1) (g : PresentedGroup rels →* G)
    (hg : ∀ x : α, g (PresentedGroup.of x) = f x) : ∀ {x}, g x = toGroup h x := by
  /-
    α : Type u_1
    G : Type u_2
    inst✝ : Group G
    f : α → G
    rels : Set (FreeGroup α)
    h : ∀ (r : FreeGroup α), Membership.mem rels r → Eq ((FreeGroup.lift f) r) 1
    g : MonoidHom (PresentedGroup rels) G
    hg : ∀ (x : α), Eq (g (PresentedGroup.of x)) (f x)
    ⊢ ∀ {x : PresentedGroup rels}, Eq (g x) ((PresentedGroup.toGroup h) x)
  -/
  intro x
  /-
    α : Type u_1
    G : Type u_2
    inst✝ : Group G
    f : α → G
    rels : Set (FreeGroup α)
    h : ∀ (r : FreeGroup α), Membership.mem rels r → Eq ((FreeGroup.lift f) r) 1
    g : MonoidHom (PresentedGroup rels) G
    hg : ∀ (x : α), Eq (g (PresentedGroup.of x)) (f x)
    x : PresentedGroup rels
    ⊢ Eq (g x) ((PresentedGroup.toGroup h) x)
  -/
  refine QuotientGroup.induction_on x ?_
  /-
    α : Type u_1
    G : Type u_2
    inst✝ : Group G
    f : α → G
    rels : Set (FreeGroup α)
    h : ∀ (r : FreeGroup α), Membership.mem rels r → Eq ((FreeGroup.lift f) r) 1
    g : MonoidHom (PresentedGroup rels) G
    hg : ∀ (x : α), Eq (g (PresentedGroup.of x)) (f x)
    x : PresentedGroup rels
    ⊢ ∀ (z : FreeGroup α), Eq (g ↑z) ((PresentedGroup.toGroup h) ↑z)
  -/
  exact fun _ ↦ FreeGroup.lift.unique (g.comp (QuotientGroup.mk' _)) hg
  /-
    🎉 no goals
  -/


@[ext]
theorem ext {φ ψ : PresentedGroup rels →* G} (hx : ∀ (x : α), φ (.of x) = ψ (.of x)) : φ = ψ := by
  /-
    α : Type u_1
    G : Type u_2
    inst✝ : Group G
    rels : Set (FreeGroup α)
    φ ψ : MonoidHom (PresentedGroup rels) G
    hx : ∀ (x : α), Eq (φ (PresentedGroup.of x)) (ψ (PresentedGroup.of x))
    ⊢ Eq φ ψ
  -/
  unfold PresentedGroup
  /-
    α : Type u_1
    G : Type u_2
    inst✝ : Group G
    rels : Set (FreeGroup α)
    φ ψ : MonoidHom (PresentedGroup rels) G
    hx : ∀ (x : α), Eq (φ (PresentedGroup.of x)) (ψ (PresentedGroup.of x))
    ⊢ Eq φ ψ
  -/
  ext
  /-
    case h.h
    α : Type u_1
    G : Type u_2
    inst✝ : Group G
    rels : Set (FreeGroup α)
    φ ψ : MonoidHom (PresentedGroup rels) G
    hx : ∀ (x : α), Eq (φ (PresentedGroup.of x)) (ψ (PresentedGroup.of x))
    a✝ : α
    ⊢ Eq ((φ.comp (QuotientGroup.mk' (Subgroup.normalClosure rels))) (FreeGroup.of …
  -/
  apply hx
  /-
    🎉 no goals
  -/


/-- Presented groups of isomorphic types are isomorphic. -/
def equivPresentedGroup (rels : Set (FreeGroup α)) (e : α ≃ β) :
    PresentedGroup rels ≃* PresentedGroup (FreeGroup.freeGroupCongr e '' rels) :=
  QuotientGroup.congr (Subgroup.normalClosure rels)
    (Subgroup.normalClosure ((FreeGroup.freeGroupCongr e) '' rels)) (FreeGroup.freeGroupCongr e)
    (Subgroup.map_normalClosure rels (FreeGroup.freeGroupCongr e).toMonoidHom
      (FreeGroup.freeGroupCongr e).surjective)


theorem equivPresentedGroup_apply_of (x : α) (rels : Set (FreeGroup α)) (e : α ≃ β) :
    equivPresentedGroup rels e (PresentedGroup.of x) =
      PresentedGroup.of (rels := FreeGroup.freeGroupCongr e '' rels) (e x) := rfl


theorem equivPresentedGroup_symm_apply_of (x : β) (rels : Set (FreeGroup α)) (e : α ≃ β) :
    (equivPresentedGroup rels e).symm (PresentedGroup.of x) =
      PresentedGroup.of (rels := rels) (e.symm x) := rfl


instance (rels : Set (FreeGroup α)) : Inhabited (PresentedGroup rels) :=
  ⟨1⟩


