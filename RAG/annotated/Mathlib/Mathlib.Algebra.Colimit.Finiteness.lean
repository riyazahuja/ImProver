/-- The directed system of finitely generated submodules of a module. -/
def fgSystem (N₁ N₂ : {N : Submodule R M // N.FG}) (le : N₁ ≤ N₂) : N₁ →ₗ[R] N₂ :=
  Submodule.inclusion le


instance : IsDirected {N : Submodule R M // N.FG} (· ≤ ·) where
  directed N₁ N₂ :=
    ⟨⟨_, N₁.2.sup N₂.2⟩, Subtype.coe_le_coe.mp le_sup_left, Subtype.coe_le_coe.mp le_sup_right⟩


instance : DirectedSystem _ (fgSystem R M · · · ·) where
  map_self _ _ := rfl
  map_map _ _ _ _ _ _ := rfl


open Submodule in
/-- Every module is the direct limit of its finitely generated submodules. -/
noncomputable def equiv : DirectLimit _ (fgSystem R M) ≃ₗ[R] M :=
  .ofBijective (lift _ _ _ _ (fun _ ↦ Submodule.subtype _) fun _ _ _ _ ↦ rfl)
    ⟨lift_injective _ _ fun _ ↦ Subtype.val_injective, fun x ↦
                                                                 /-
                                                                   R : Type u_1
                                                                   M : Type u_2
                                                                   inst✝³ : Semiring R
                                                                   inst✝² : AddCommMonoid M
                                                                   inst✝¹ : Module R M
                                                                   inst✝ : DecidableEq (Submodule R M)
                                                                   x : M
                                                                   ⊢ Membership.mem (Singleton.singleton x) x
                                                                 -/
      ⟨of _ _ _ _ ⟨_, fg_span_singleton x⟩ ⟨x, subset_span <| by rfl⟩, lift_of ..⟩⟩
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


lemma equiv_comp_of (N : {N : Submodule R M // N.FG}) :
    (equiv R M).toLinearMap ∘ₗ of _ _ _ _ N = N.1.subtype := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝³ : Semiring R
    inst✝² : AddCommMonoid M
    inst✝¹ : Module R M
    inst✝ : DecidableEq (Submodule R M)
    N : Subtype fun N => N.FG
    ⊢ Eq ((↑(Module.fgSystem.equiv R M)).comp (Module.DirectLimit.of R (Subtype fu …
  -/
  ext; simp [equiv]
       /-
         🎉 no goals
       -/


