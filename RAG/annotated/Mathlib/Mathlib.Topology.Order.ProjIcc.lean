protected theorem Filter.Tendsto.IccExtend (f : γ → Icc a b → β) {la : Filter α} {lb : Filter β}
    {lc : Filter γ} (hf : Tendsto (↿f) (lc ×ˢ la.map (projIcc a b h)) lb) :
    Tendsto (↿(IccExtend h ∘ f)) (lc ×ˢ la) lb :=
  hf.comp <| tendsto_id.prod_map tendsto_map


@[continuity]
theorem continuous_projIcc : Continuous (projIcc a b h) :=
  (continuous_const.max <| continuous_const.min continuous_id).subtype_mk _


theorem isQuotientMap_projIcc : IsQuotientMap (projIcc a b h) :=
  isQuotientMap_iff.2 ⟨projIcc_surjective h, fun s =>
                                                                    /-
                                                                      α : Type u_1
                                                                      inst✝² : LinearOrder α
                                                                      a b : α
                                                                      h : LE.le a b
                                                                      inst✝¹ : TopologicalSpace α
                                                                      inst✝ : OrderTopology α
                                                                      s : Set ↑(Set.Icc a b)
                                                                      hs : IsOpen (Set.preimage (Set.projIcc a b h) s)
                                                                      ⊢ Eq (Set.preimage Subtype.val (Set.preimage (Set.projIcc a b h) s)) s
                                                                    -/
    ⟨fun hs => hs.preimage continuous_projIcc, fun hs => ⟨_, hs, by ext; simp⟩⟩⟩
                                                                         /-
                                                                           🎉 no goals
                                                                         -/


@[deprecated (since := "2024-10-22")]
alias quotientMap_projIcc := isQuotientMap_projIcc


@[simp]
theorem continuous_IccExtend_iff {f : Icc a b → β} : Continuous (IccExtend h f) ↔ Continuous f :=
  isQuotientMap_projIcc.continuous_iff.symm


/-- See Note [continuity lemma statement]. -/
protected theorem Continuous.IccExtend {f : γ → Icc a b → β} {g : γ → α} (hf : Continuous ↿f)
    (hg : Continuous g) : Continuous fun a => IccExtend h (f a) (g a) :=
  show Continuous (↿f ∘ fun x => (x, projIcc a b h (g x)))
  from hf.comp <| continuous_id.prod_mk <| continuous_projIcc.comp hg


/-- A useful special case of `Continuous.IccExtend`. -/
@[continuity]
protected theorem Continuous.Icc_extend' {f : Icc a b → β} (hf : Continuous f) :
    Continuous (IccExtend h f) :=
  hf.comp continuous_projIcc


theorem ContinuousAt.IccExtend {x : γ} (f : γ → Icc a b → β) {g : γ → α}
    (hf : ContinuousAt (↿f) (x, projIcc a b h (g x))) (hg : ContinuousAt g x) :
    ContinuousAt (fun a => IccExtend h (f a) (g a)) x :=
  show ContinuousAt (↿f ∘ fun x => (x, projIcc a b h (g x))) x from
    ContinuousAt.comp hf <| continuousAt_id.prod <| continuous_projIcc.continuousAt.comp hg

